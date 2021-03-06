import random
import gym
import numpy as np
import tensorflow as tf
import pprint
import tempfile
import os
import shutil
import json

from collections import defaultdict
from gym import wrappers

MAX_STEPS_PER_EPISODE = 200
NUM_EPISODES = 150
OPEN_AI_KEY = os.environ.get('OPEN_AI_KEY')

pp = pprint.PrettyPrinter()


def print_memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', memoryUse)


class QModel2(object):

    def __init__(self, sess, trainer, N1=100, N2=50, input_dim=4, actions_dim=2):
        self.sess = sess
        self.trainer = trainer
        self.input_dim = input_dim
        self.actions_dim = actions_dim

        minv = -.001
        maxv = .001
        W1 = tf.Variable(tf.random_uniform([self.input_dim, N1], minval=minv, maxval=maxv))
        b1 = tf.Variable(tf.random_uniform([N1], minval=minv, maxval=maxv))
        W2 = tf.Variable(tf.random_uniform([N1, N2], minval=minv, maxval=maxv))
        b2 = tf.Variable(tf.random_uniform([N2], minval=minv, maxval=maxv))
        W3 = tf.Variable(tf.random_uniform([N2, self.actions_dim], minval=-1, maxval=1))
        b3 = tf.Variable(tf.random_uniform([self.actions_dim], minval=-1, maxval=1))

        # Make Q and Q'
        self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')
        L1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
        self.output = tf.matmul(L2, W3) + b3
        self.actions = tf.placeholder(tf.int32, [None, 2], name='actions')
        self.q = tf.gather_nd(self.output, self.actions)

        self.x_ = tf.placeholder(tf.float32, [None, self.input_dim], name='x_')
        L1_ = tf.nn.relu(tf.matmul(self.x_, W1) + b1)
        L2_ = tf.nn.relu(tf.matmul(L1_, W2) + b2)
        self.output_ = tf.matmul(L2_, W3) + b3
        self.q_ = tf.reduce_max(self.output_, reduction_indices=[1], name='x_')

        self.r = tf.placeholder(tf.float32, [None], name='r')
        self.pred = self.q
        self.actual = self.r + self.q_

        self.loss = .5 * tf.pow(self.actual - self.pred, 2)
        self.train_step = self.trainer.minimize(tf.reduce_mean(self.loss))
        # self.sess.graph.finalize()

    def suggest(self, batch_x):
        out = self.sess.run(self.output, feed_dict={self.x: batch_x})
        s = np.argmax(out, axis=1)  # TODO: don't need numpy here
        return s

    def train(self, batch_x, batch_actions, batch_x_, batch_r):
        fd = {
            self.x: batch_x,
            self.actions: batch_actions,
            self.x_: batch_x_,
            self.r: batch_r,
        }

        # if not self.did_init_adam:
        #     tf.global_variables_initializer().run()  # For AdamOptimizer
        #     self.did_init_adam = True
        self.sess.run(self.train_step, feed_dict=fd)


class CartPole():
    def __init__(self, env, model, render=False):
        self.env = env
        self.model = model
        self.render = render
        self.memory = []

    def run_episode(self):
        obs = self.env.reset()
        for i in range(MAX_STEPS_PER_EPISODE):
            if self.render:
                self.env.render()
            action = self.model.suggest([obs])[0]
            new_obs, reward, done, _ = self.env.step(action)
            real_reward = int(not done)
            new_memory = (obs, action, new_obs, float(real_reward))
            self.memory.append(new_memory)
            obs = new_obs
            if done:
                break
        return i

    def run_episodes(self, n):
        best = 0
        for _ in range(n):
            best = max(best, self.run_episode())
        return best

    def build_memory(self, target_size):
        best = 0
        while len(self.memory) < target_size:
            n = self.run_episode()
            if n > best:
                best = n
        print 'best was', best
        return best

    def pop_memory(self):
        m = self.memory
        self.memory = []
        return m

    def pull_memory(self, n):
        memory = []
        for _ in range(n):
            memory.append(self.memory[random.randint(0, len(self.memory) - 1)])
        return memory

    def forget(self, percent):
        n = int(percent * len(self.memory))
        if n == 0:
            self.memory = []
            return
        for _ in range(n):
            idx = random.randint(0, len(self.memory) - 1)
            del[idx]


def permute_params(base_params):
    num_perms = 1
    for key in base_params.keys():
        num_perms *= len(base_params[key])

    perms = []
    for i in range(num_perms):
        perm = {}
        div_by = 1
        for j in range(len(base_params.keys())):
            key = base_params.keys()[j]
            idx = i/div_by % len(base_params[key])
            div_by *= len(base_params[key])
            perm[key] = base_params[key][idx]
        perms.append(perm)
    return perms


def pprint_perm_results(results):
    keys = results.keys()
    keys.sort(key=lambda x: sum(results[x]))
    for key in keys:
        r = results[key]
        print '- %.01f: %s - %s...%s' % (float(sum(r))/len(r), key, r[:10], r[-10:-1])

    results_by_value = defaultdict(lambda: {'sum': 0, 'count': 0})
    for perm_json, r in results.iteritems():
        perm = json.loads(perm_json)
        for param, value in perm.iteritems():
            kv = '%s=%s' % (param, value)
            results_by_value[kv]['sum'] += sum(r)
            results_by_value[kv]['count'] += len(r)

    print 'Avg per hyperparam value:'
    for kv in sorted(results_by_value.keys()):
        score = results_by_value[kv]
        print '- %s: %01.f' % (kv, float(score['sum'])/score['count'])
    print ''


def run_with_params(N, base_params, upload=False):
    perms = permute_params(base_params)
    pp.pprint(perms)

    results = {}
    for perm_i, perm in enumerate(perms):
        print_memory()
        print '== Permutation %i of %i: %s ==' % (perm_i, len(perms), perm)
        result = run_param_permutation(N, perm, upload)
        results[json.dumps(perm)] = result
        print 'Results so far:'
        pprint_perm_results(results)

    return results


def run_param_permutation(N, perm, upload=False):
    env = gym.make('CartPole-v0')
    if upload:
        data_path = '%s/cartpole' % tempfile.gettempdir()
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        env = wrappers.Monitor(env, data_path)
    with tf.Graph().as_default(), tf.Session() as sess:
        trainer = tf.train.AdamOptimizer(
            perm['learning_rate'],
            beta1=perm['beta1'],
            beta2=perm['beta2'])
        model = QModel2(sess, trainer, N1=perm['N1'], N2=perm['N2'])
        sess.run(tf.global_variables_initializer())

        cp = CartPole(env, model, render=False)
        result = []
        for i in range(N):
            steps = cp.run_episodes(1)
            memory = cp.pull_memory(1000)
            result.append(steps)
            cp.forget(.1)
            if (i % 50 == 0):
                print '%s/%s) steps: %s' % (i, N, steps)
            batch_x = [i[0] for i in memory]
            batch_actions = [[i, x[1]] for i, x in enumerate(memory)]
            batch_x_ = [i[2] for i in memory]
            batch_r = [i[3] for i in memory]
            model.train(batch_x, batch_actions, batch_x_, batch_r)
    env.close()
    if upload:
        gym.upload(data_path, api_key=OPEN_AI_KEY)
    tf.reset_default_graph()
    return result


if __name__ == '__main__':
    # base_params = {
    #     'learning_rate': [.1, .05, .02, .01],
    #     'beta1': [.8, .9, .95],
    #     'beta2': [.95, .99, .999],
    #     'N1': [10, 50, 100, 500],
    #     'N2': [10, 50, 100, 500],
    # }
    base_params = {
        'learning_rate': [.01, .02, .05],
        'beta1': [.95],
        'beta2': [.95],
        'N1': [100, 500],
        'N2': [50, 100],
    }
    results = run_with_params(1000, base_params, upload=True)
    print 'Results:'
    pprint_perm_results(results)

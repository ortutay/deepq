import random
import gym
import numpy as np
import tensorflow as tf
import pprint

MAX_STEPS_PER_EPISODE = 200
NUM_EPISODES = 150

pp = pprint.PrettyPrinter()


class QModel2(object):

    def __init__(self, sess, trainer, input_dim=4, actions_dim=2):
        self.sess = sess
        self.trainer = trainer
        self.input_dim = input_dim
        self.actions_dim = actions_dim

        N1 = 100
        N2 = 50

        # TODO: think about making this symmetric with minval=-1
        minv = -.001
        maxv = .001
        W1 = tf.Variable(tf.random_uniform([self.input_dim, N1], minval=minv, maxval=maxv))
        b1 = tf.Variable(tf.random_uniform([N1], minval=minv, maxval=maxv))
        W2 = tf.Variable(tf.random_uniform([N1, N2], minval=minv, maxval=maxv))
        b2 = tf.Variable(tf.random_uniform([N2], minval=minv, maxval=maxv))

        W3 = tf.Variable(tf.random_uniform([N2, self.actions_dim], minval=-1, maxval=1))
        b3 = tf.Variable(tf.random_uniform([self.actions_dim], minval=-1, maxval=1))

        # Make Q and Q'
        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        L1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
        self.output = tf.matmul(L2, W3) + b3
        self.q = tf.reduce_max(self.output, reduction_indices=[1])

        self.x_ = tf.placeholder(tf.float32, [None, self.input_dim])
        L1_ = tf.nn.relu(tf.matmul(self.x_, W1) + b1)
        L2_ = tf.nn.relu(tf.matmul(L1_, W2) + b2)
        self.output_ = tf.matmul(L2_, W3) + b3
        self.q_ = tf.reduce_max(self.output_, reduction_indices=[1])

        self.r = tf.placeholder(tf.float32, [None])
        self.pred = self.q
        self.actual = self.r + self.q_

        self.loss = .5 * tf.pow(self.actual - self.pred, 2)

    def suggest(self, batch_x):
        # s = self.sess.run(tf.argmax(self.output), feed_dict={self.x: batch_x})
        out = self.sess.run(self.output, feed_dict={self.x: batch_x})
        s = np.argmax(out, axis=1)
        # print 'batch x', batch_x
        # print 'move', out, s
        if random.random() < .1:
            return [random.randint(0, 1)]
        else:
            return s

    def train(self, batch_x, batch_x_, batch_r):
        fd = {
            self.x: batch_x,
            self.x_: batch_x_,
            self.r: batch_r,
        }
        # print 'fd', fd
        # pred = sess.run(self.pred, feed_dict=fd)
        # actual = sess.run(self.actual, feed_dict=fd)
        loss = sess.run(tf.reduce_mean(self.loss), feed_dict=fd)
        # print 'loss, pred, actual', loss, pred, actual
        print 'loss', loss
        train_step = self.trainer.minimize(tf.reduce_mean(self.loss))
        sess.run(train_step, feed_dict=fd)


class CartPole():
    def __init__(self, env, model, render=False):
        self.env = env
        self.model = model
        self.render = render
        self.memory = []

    def run_episode(self):
        obs = self.env.reset()
        for i in range(MAX_STEPS_PER_EPISODE):
            # print 'step', i
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

    def build_memory(self, target_size):
        best = 0
        while len(self.memory) < target_size:
            n = self.run_episode()
            if n > best:
                best = n
        print 'best was', best

    def pop_memory(self):
        m = self.memory
        self.memory = []
        return m

    # def train_q(target_episodes=100):
    #     model = QModel()
    #     for i in range(NUM_EPISODES):
    #         # TODO: train model?
    #         steps = run_episode(env, model, render)
    #         if steps >= target_episodes:
    #             return i
    #     print 'i failed'
    #     return i

    # def run_and_plot(use, number_of_runs=100, render=False):
    #     env = gym.make('CartPole-v0')
    #     num_episodes = []
    #     print 'Using {}'.format(use)
    #     target_episodes = 199
    #     for i in range(number_of_runs):
    #         num_episodes.append(globals()[use](env, target_episodes, render=render))
    #     success_num_episodes = []
    #     for x in num_episodes:
    #         if x < target_episodes / 2:
    #             success_num_episodes.append(x)
    #     print 'all episodes', num_episodes
    #     print 'succcess episodes', success_num_episodes
    #     print use, np.mean(success_num_episodes)
    #     import matplotlib.pyplot as plt
    #     plt.title(use)
    #     plt.hist(num_episodes)
    #     plt.show()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    sess = tf.InteractiveSession()
    # trainer = tf.train.AdamOptimizer(.05, beta1=.9, beta2=.999)
    trainer = tf.train.GradientDescentOptimizer(.05)
    model = QModel2(sess, trainer)
    tf.global_variables_initializer().run()

    cp = CartPole(env, model, render=False)
    N = 1000
    for i in range(N):
        # cp.run_episode()
        cp.build_memory(10000)
        memory = cp.pop_memory()
        print '%s/%s) steps: %s' % (i, N, len(memory))
        batch_x = [i[0] for i in memory]
        batch_x_ = [i[2] for i in memory]
        batch_r = [i[3] for i in memory]
        model.train(batch_x, batch_x_, batch_r)

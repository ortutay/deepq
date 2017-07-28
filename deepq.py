import gym
import numpy as np
import tensorflow as tf
import pprint

STEPS_PER_EPISODE = 200
NUM_EPISODES = 150

pp = pprint.PrettyPrinter()


class QModel(object):

    def __init__(self, sess):
        self.sess = sess

        self.x = tf.placeholder(tf.float32, [None, 4])
        # TODO: think about making this symmetric with minval=-1
        self.W1 = tf.Variable(tf.random_uniform([4, 100]))
        self.b1 = tf.Variable(tf.random_uniform([100]))
        self.W2 = tf.Variable(tf.random_uniform([100, 2]))
        self.b2 = tf.Variable(tf.random_uniform([2]))

        self.L1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)
        self.output = tf.matmul(self.L1, self.W2) + self.b2

    def train(self, batch_x, batch_u, batch_x_, batch_y):
        r = tf.placeholder(tf.float32, [None, 1])
        q = tf.gather(self.output, batch_u)
        # x_ = tf.placeholder(tf.float32, [None, 4])
        output_ = self.eval(batch_x) # Q(s', a')
        q_ = tf.reduce_max(output_, axis=1)
        # import pdb; pdb.set_trace()
        loss = tf.pow((r + q_ - q), 2)
        train_step = tf.train.GradientDescentOptimizer(.05).minimize(loss)
        sess.run(train_step, feed_dict={
            self.x: batch_x,
            x_: batch_x_,
            r: batch_y,
        })

    def eval(self, batch_x):
        return self.sess.run(self.output, feed_dict={self.x: batch_x})

    def suggest(self, batch_x):
        return np.argmax(self.eval(batch_x), axis=1)


class CartPole():
    def __init__(self, env, model, render=False):
        self.env = env
        self.model = model
        self.render = render
        self.memory = []

    def run_episode(self):
        obs = self.env.reset()
        for i in range(STEPS_PER_EPISODE):
            print 'step', i
            if self.render:
                self.env.render()
            action = self.model.suggest([obs])[0]
            new_obs, reward, done, _ = self.env.step(action)
            real_reward = int(not done)
            new_memory = (obs, action, new_obs, float(real_reward))
            self.memory.append(new_memory)
            if done:
                break
        return i

    def build_memory(self, target_size):
        while len(self.memory) < target_size:
            self.run_episode()

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
    model = QModel(sess)
    tf.global_variables_initializer().run()

    cp = CartPole(env, model, render=True)
    cp.run_episode()
    # cp.build_memory(100)

    print 'here is my memory'
    pp.pprint(cp.memory)
    batch_x = [i[0] for i in cp.memory]
    batch_action = [i[1] for i in cp.memory]
    batch_x_ = [i[2] for i in cp.memory]
    batch_y = [i[3] for i in cp.memory]
    print 'train it!'
    model.train(batch_x, batch_action, batch_x_, batch_y)

    # run_and_plot('train_q', render=True)
    # sess = tf.InteractiveSession()
    # model = QModel(sess)
    # tf.global_variables_initializer().run()

    # batch_x = [
    #     [1, 2, .5, -.23],
    #     [12, 2, .5, -.23],
    #     [13, 2, .5, -.23],
    #     [14, 2, .5, -.23],
    #     [-1016, 21, .5, -.23],
    #     [1, 23, .5, -.23],
    #     [1, 24, .5, -.23],
    # ]
    # y = model.eval(batch_x)
    # print y
    # act = model.suggest(batch_x)
    # print act

import gym
import numpy as np
import tensorflow as tf

STEPS_PER_EPISODE = 200
NUM_EPISODES = 150


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

    def eval(self, batch_x):
        return self.sess.run(self.output, feed_dict={self.x: batch_x})


def run_episode(env, model, render=False):
    observation = env.reset()
    for i in range(STEPS_PER_EPISODE):
        if render:
            env.render()
        action = model.eval(observation)
        observation, reward, done, _ = env.step(action)
        if done:
            break
    return i


def train_q(env, target_episodes=100, render=False):
    model = QModel()
    for i in range(NUM_EPISODES):
        # TODO: train model?
        steps = run_episode(env, model, render)
        if steps >= target_episodes:
            return i
    print 'i failed'
    return i


def run_and_plot(use, number_of_runs=100, render=False):
    env = gym.make('CartPole-v0')
    num_episodes = []
    print 'Using {}'.format(use)
    target_episodes = 199
    for i in range(number_of_runs):
        num_episodes.append(globals()[use](env, target_episodes, render=render))
    success_num_episodes = []
    for x in num_episodes:
        if x < target_episodes / 2:
            success_num_episodes.append(x)
    print 'all episodes', num_episodes
    print 'succcess episodes', success_num_episodes
    print use, np.mean(success_num_episodes)
    import matplotlib.pyplot as plt
    plt.title(use)
    plt.hist(num_episodes)
    plt.show()


if __name__ == '__main__':
    # run_and_plot('train_q', render=True)
    sess = tf.InteractiveSession()
    model = QModel(sess)
    tf.global_variables_initializer().run()

    y = model.eval([
        [1, 2, .5, -.23],
        [12, 2, .5, -.23],
        [13, 2, .5, -.23],
        [14, 2, .5, -.23],
        [16, 21, .5, -.23],
        [1, 23, .5, -.23],
        [1, 24, .5, -.23],
    ])
    print y

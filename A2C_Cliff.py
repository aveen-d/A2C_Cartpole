

import gym
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import collections

if "../" not in sys.path:
  sys.path.append("../") 
from Cliffwalking_env import CliffWalkingEnv



en1 = CliffWalkingEnv()

class policyest():

    
    def __init__(self, lr=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.st = tf.placeholder(tf.int32, [], "state")
            self.axn = tf.placeholder(dtype=tf.int32, name="action")
            self.trgt = tf.placeholder(dtype=tf.float32, name="target")

            # Its a lookup table estimator
            st_oneh = tf.one_hot(self.st, int(en1.observation_space.n))
            self.outL = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(st_oneh, 0),
                num_outputs=en1.action_space.n,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.axn_prob = tf.squeeze(tf.nn.softmax(self.outL))
            self.picked_action_prob = tf.gather(self.axn_prob, self.axn)

            # calculating loss and training
            self.loss = -tf.log(self.picked_action_prob) * self.trgt

            self.optmzr = tf.train.AdamOptimizer(learning_rate=lr)
            self.trainingOP = self.optmzr.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def pred(self, st, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.axn_prob, { self.st: st })

    def upd(self, st, target, axn, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.st: st, self.trgt: target, self.axn: axn  }
        _, loss = sess.run([self.trainingOP, self.loss], feed_dict)
        return loss

class value_est():

    def __init__(self, lr=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.st = tf.placeholder(tf.int32, [], "state")
            self.trgt = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            st_oneh = tf.one_hot(self.st, int(en1.observation_space.n))
            self.outL = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(st_oneh, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.outL)
            self.loss = tf.squared_difference(self.value_estimate, self.trgt)

            self.optmzr = tf.train.AdamOptimizer(learning_rate=lr)
            self.trainingOP = self.optmzr.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def pred(self, st, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.st: st })

    def upd(self, st, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.st: st, self.trgt: target }
        _, loss = sess.run([self.trainingOP, self.loss], feed_dict)
        return loss
      
def actor_critic(en1, pol_est, val_est, n_ep, discount_factor=1.0):

    ep_len=np.zeros(num_episodes)
	ep_reward=np.zeros(num_episodes)
	Q=[]   
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    for i_episode in range(n_ep):
        # pick the first action after resetting the environment
        st = en1.reset()
        
        episode = []
        
        # taking a single step in environment
        for t in itertools.count():
            
         
            axn_prob = pol_est.pred(st)
            axn = np.random.choice(np.arange(len(axn_prob)), p=axn_prob)
            next_state, reward, done, _ = en1.step(axn)
            
            # tracking transition
            episode.append(Transition(
              st=st, action=axn, reward=reward, next_state=next_state, done=done))
            
            ep_len[i_episode]=t
			ep_reward[i_episode] += reward
            
            # TD Target calculation
            value_next = val_est.pred(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - val_est.pred(st)
            
            #value estimator is updated
            val_est.upd(st, td_target)
            
            # policy estimator is updated using td error as advantage estimation
            pol_est.upd(st, td_error, axn)
            
          
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, n_ep, ep_reward[i_episode - 1]), end="")

            if done:
                break
                
            st = next_state
    
    return (ep_len,ep_reward,Q)
  
tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
pol_est = policyest()
val_est = value_est()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    ep_len,ep_reward,Q=actor_critic(env, policy_estimator, value_estimator, 300)

	plt.plot(Q,ep_reward,label='Reward vs Episodes',color='b')
	plt.show()
	plt.plot(Q,ep_len,label='EPisode lenght vs Episode',color='b')
	plt.show()
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import deeprl_hw3.reinforce as reinforce

def main():
	env = gym.make('CartPole-v0')
	with tf.Session() as sess:
		reward = reinforce.reinforce(env, sess)
		print(reward)

if __name__ == '__main__':
	main()
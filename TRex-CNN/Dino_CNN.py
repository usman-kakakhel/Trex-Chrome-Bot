import random
import numpy as np
from collections import deque
from Dino_Chrome_env import Dino_env

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential

def build_model(sliding_input_shape=(150,300,4)):
	model = Sequential([
			Input(shape=sliding_input_shape),
			Conv2D(32, (3, 3), activation='relu', padding='same'),
			MaxPooling2D(),
			Conv2D(64, (3, 3), activation='relu', padding='same'),
			MaxPooling2D(),
			Conv2D(128, (3, 3), activation='relu', padding='same'),
			MaxPooling2D(),
			Conv2D(256, (3, 3), activation='relu', padding='same'),
			MaxPooling2D(),
			Dropout(0.7),
			Conv2D(512, (3, 3), activation='relu', padding='same'),
			MaxPooling2D(),
			Dropout(0.7),
			Conv2D(1024, (3, 3), activation='relu', padding='same'),
			MaxPooling2D(),
			Dropout(0.7),
			Flatten(),
			Dense(512, activation='relu'),
			Dense(64, activation="relu"),
			Dense(8, activation="relu"),
			Dense(2, activation="linear"),
		])
	nadam = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
	model.compile(optimizer=nadam, loss='mse')
	return model

class GameAgent:
	def __init__(self, gamma=0.99, epsilon=0.5, alpha=0.9, batchSize=256, 
				maxBufferSize=35000, minEpsilon=0.00001, decayEpsilon=0.999):
		self.epsilon = epsilon
		self.minEpsilon = minEpsilon
		self.decayEpsilon = decayEpsilon
		self.batchSize = batchSize
		self.gamma = gamma
		self.alpha = alpha
		self.maxBufferSize = maxBufferSize
		self.memory = deque(maxlen=maxBufferSize)
		self.bufferItr = 0
		self.overflow = False
		self.brain = build_model()

	def storeMem(self, state, action, reward, nextState, done):
		if self.bufferItr >= self.maxBufferSize:
			self.bufferItr = 0
			self.overflow = True

		if self.overflow:
			self.bufferItr = self.bufferItr % self.maxBufferSize
			self.memory[self.bufferItr] = [state, action, reward, nextState, done]
			self.bufferItr += 1
		else:
			self.memory.append([state, action, reward, nextState, done])
			self.bufferItr += 1

	def act(self, state):
		if np.random.rand(1) < self.epsilon:
			print("rand")
			return np.random.randint(0, 2)
		else:
			print("nott")
			return tf.argmax(self.brain(state), axis=1).numpy().squeeze()

	def decay(self):
		if self.epsilon > self.minEpsilon:
			self.epsilon = self.epsilon * self.decayEpsilon
		else:
			self.epsilon = self.minEpsilon

	def train(self):
		if len(self.memory) < self.batchSize:
			return

		batch = random.sample(self.memory, self.batchSize)
		npBatch = np.array(batch, dtype=object)

		statesTemp, actionsTemp, rewardsTemp, newstatesTemp, donesTemp = np.hsplit(npBatch, 5)
		states = np.concatenate((np.squeeze(statesTemp[:])), axis = 0)
		newstates = np.concatenate(np.concatenate(newstatesTemp))
		actions = actionsTemp.reshape(self.batchSize,).astype(int)
		rewards = rewardsTemp.reshape(self.batchSize,).astype(float)
		dones = np.concatenate(donesTemp).astype(bool)
		notdones = ~dones
		notdones = notdones.astype(float)
		dones = dones.astype(float)
		
		qValue = self.brain(states).numpy()
		maxCurrQValue = np.amax(qValue, axis=1)
		maxNextQValue = np.amax(self.brain(newstates).numpy(), axis=1)
		targetQValue = qValue.copy()

		indexes = np.arange(self.batchSize)
		a = rewards * dones
		b = (maxCurrQValue + self.alpha * (rewards + maxNextQValue * self.gamma - maxCurrQValue)) * notdones
		targetQValue[(indexes, actions)] = a * b

		history = self.brain.fit(states, targetQValue, batch_size=16, epochs=1, verbose=0)

if __name__ == "__main__":
	myEnv = Dino_env()
	dqn = GameAgent()

	myEnv.step(1)
	eps = 1
	while True:
		st, reward, done_ = myEnv.step(0)
		st = np.stack((st, st, st, st), axis=2)
		currentState = st.reshape(1, st.shape[0], st.shape[1], st.shape[2])
		rewardSum = 0
		curr_score = myEnv.getScore()
		high_score = myEnv.getHighScore()
		done = False

		while not done:
			bestAction = dqn.act(currentState)
			new_st, reward, done = myEnv.step(bestAction)
			new_st = new_st.reshape(1, new_st.shape[0], new_st.shape[1], 1)
			new_state = np.append(currentState[:, :, :, 1:], new_st, axis=3)
			if myEnv.getScore() > curr_score:
				curr_score = myEnv.getScore()

			dqn.storeMem(currentState, bestAction, reward, new_state, done)

			dqn.train()
			rewardSum += reward
			currentState = new_state
			dqn.decay()

		print("Reached score: {} in episode: {} with high score: {}, epsilon: {}".format(curr_score, eps, myEnv.getHighScore(), dqn.epsilon))
		eps += 1
		if myEnv.getHighScore() > high_score:
			high_score = myEnv.getHighScore()

	myEnv.stop()
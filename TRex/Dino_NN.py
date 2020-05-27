import dino_env as env
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import time

eps = 0

class Brain(nn.Module):
	def __init__(self, inputSize, h1, h2, h3, h4, h5, outputSize, lr):
		super(Brain, self).__init__()
		self.inp = inputSize
		self.out = outputSize
		self.h1 = h1
		self.h2 = h2
		self.h3 = h3
		self.h4 = h4
		self.h5 = h5

		self.hiddenLayer1 = nn.Linear(self.inp, self.h1)
		self.hiddenLayer2 = nn.Linear(self.h1, self.h2)
		self.hiddenLayer3 = nn.Linear(self.h2, self.h3)
		self.hiddenLayer4 = nn.Linear(self.h3, self.h4)
		self.hiddenLayer5 = nn.Linear(self.h4, self.h5)
		self.output = nn.Linear(self.h5, self.out)
		
		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		self.lossFunc = nn.MSELoss()
		self.device = torch.device('cpu')
		self.to(self.device)
	

	def forward(self, x):
		scaler = MinMaxScaler(feature_range=(0,1))
		x = scaler.fit_transform(x)
		x = torch.Tensor(x).to(self.device)
		x = F.relu(self.hiddenLayer1(x))
		x = F.relu(self.hiddenLayer2(x))
		x = F.relu(self.hiddenLayer3(x))
		x = F.relu(self.hiddenLayer4(x))
		x = F.relu(self.hiddenLayer5(x))
		return self.output(x)
		
class GameAgent():
	def __init__(self, gamma, epsilon, alpha, batchSize, inputSize, h1Size, 
				h2Size, h3Size, h4Size, h5Size, outputSize, maxBufferSize=50000, minEpsilon=0.001, decayEpsilon=0.9995):
		global eps
		self.epsilon = epsilon
		self.minEpsilon = minEpsilon
		self.decayEpsilon = decayEpsilon
		self.batchSize = batchSize
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.gamma = gamma
		self.maxBufferSize = maxBufferSize
		self.brain = Brain(inputSize, h1Size, h2Size, h3Size, h4Size, h5Size, outputSize, alpha)
		state_dict, self.memory, self.bufferItr, eps = env.control.loader("Logs/")
		if state_dict is not None:
			self.brain.load_state_dict(state_dict)
			self.brain.eval()
			self.epsilon = self.minEpsilon
		

	def storeMem(self, state, action, reward, nextState, done):
		state = state.reshape((1, self.inputSize))
		nextState = nextState.reshape((1, self.inputSize))
		if self.bufferItr >= self.maxBufferSize:
			self.bufferItr = self.bufferItr % self.maxBufferSize
			self.memory[self.bufferItr] = [state, action, reward, nextState, done]
			self.bufferItr += 1
		else:
			self.bufferItr += 1
			self.memory.append([state, action, reward, nextState, done])

	def act(self, state):
		if np.random.rand(1) < self.epsilon:
			return np.random.randint(0, self.outputSize)
		else:
			return self.brain.forward(state.reshape((1, self.inputSize))).argmax().item()

	def decay(self):
		if self.epsilon > self.minEpsilon:
			self.epsilon = self.epsilon * self.decayEpsilon
		else:
			self.epsilon = self.minEpsilon

	def train(self):
		if len(self.memory) < self.batchSize:
			return
		
		batch = random.sample(self.memory, self.batchSize)
		npBatch = np.array(batch)

		statesTemp, actionsTemp, rewardsTemp, newstatesTemp, donesTemp = np.hsplit(npBatch, 5)

		states = np.concatenate((np.squeeze(statesTemp[:])), axis = 0)
		newstates = np.concatenate(np.concatenate(newstatesTemp))

		rewards = rewardsTemp.reshape(self.batchSize,).astype(float)
		dones = np.concatenate(donesTemp).astype(bool)
		notdones = ~dones
		notdones = notdones.astype(float)
		dones = dones.astype(float)

		qValue = self.brain.forward(states)
		targetQValue = qValue.clone().cpu().detach().numpy()
		nextQValue = self.brain.forward(newstates)
		npNextQValue = np.amax(nextQValue.clone().cpu().detach().numpy(), axis=1)
		
		actions = actionsTemp.reshape(self.batchSize,).astype(int)
		indexes = np.arange(self.batchSize)
		targetQValue[(indexes, actions)] = rewards * dones + (rewards + npNextQValue * self.gamma) * notdones
		targetQValue = torch.Tensor(targetQValue).to(self.brain.device)

		self.brain.optimizer.zero_grad()
		loss = self.brain.lossFunc(qValue, targetQValue)
		loss.backward()
		self.brain.optimizer.step()
		
myEnv = env.DinoEnv()
dqn = GameAgent(0.995, 1.0, 0.0001, 24, env.OBSERVATION_SPACE, 1024, 1024, 1024, 1024, 1024,env.ACTION_SPACE)
stat_file = open("stats.txt","a")
counter = 0

while True:
	currentState = np.array(myEnv.start())
	rewardSum = 0
	curr_score = env.getScore()
	done = False

	while not done:
		env.render()
		bestAction = dqn.act(currentState)
		new_state, reward, done = env.step(bestAction)
		
		if env.getScore() > curr_score:
			curr_score = env.getScore()

		dqn.storeMem(currentState, bestAction, reward, np.asarray(new_state), done)
		
		if counter*24 >= 50000:
			env.control.saver(dqn.brain.state_dict(),dqn.memory,dqn.bufferItr, eps, "Logs/", True)
			counter = 0
		else:
			counter += 1

		dqn.train()
		rewardSum += reward
		currentState = np.asarray(new_state)
		dqn.decay()
		env.control.event_loop(dqn.brain.state_dict(),dqn.memory,dqn.bufferItr, eps, "Logs/", dqn.epsilon <= dqn.minEpsilon)

	stat_file.write(str(eps) + ", " + str(curr_score) + "\n")
	print("Reached score: {} in episode: {} with high score: {}".format(curr_score, eps, env.high_score))
	eps += 1

myEnv.stop()

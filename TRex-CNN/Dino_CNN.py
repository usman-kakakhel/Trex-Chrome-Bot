import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import random
import numpy as np 
import time
from Dino_Chrome_env import Dino_env
import os
import pickle

eps = 0

def save_obj(obj, path, name):
    with open(path + name + '.pkl', 'wb') as f: #dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
		
def load_obj(path, name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def saver(state_dict, mem, itr, eps, path, toBeSaved):
	if toBeSaved:
		torch.save(state_dict, path + "model.pt")
		save_obj(mem, path, 'mem')
		save_obj(itr, path, 'itr')
		save_obj(eps, path, 'eps')

def loader(path):
	if os.path.exists(path + "model.pt"):
		state_dict = torch.load(path + "model.pt")
	else:
		state_dict = None
	if os.path.exists(path + "mem.pkl"):
		mem = load_obj(path, 'mem')
	else:
		maxBufferSize=100000
		mem = deque(maxlen=maxBufferSize)
	if os.path.exists(path + "itr.pkl"):
		itr = load_obj(path, 'itr')
	else:
		itr = 0
	if os.path.exists(path + "eps.pkl"):
		eps = load_obj(path, 'eps')
	else:
		eps = 0
	return state_dict, mem, itr, eps

class Brain(nn.Module):
	def __init__(self, lr):
		super(Brain, self).__init__()
		self.h1 = nn.Conv2d(4, 32, 8, 4)
		self.pool1 = nn.MaxPool2d(2)
		self.h2 = nn.Conv2d(32, 64, 4, 1)
		self.h3 = nn.Conv2d(64, 64, 3, 1)
		self.h4 = nn.Linear(1600, 256)
		self.h5 = nn.Linear(256 , 2)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		self.lossFunc = nn.MSELoss()
		self.apply(init_weights)
	

	def forward(self, x):
		x = torch.Tensor(x)
		x = F.relu(self.h1(x))
		x = self.pool1(x)
		x = F.relu(self.h2(x))
		x = F.relu(self.h3(x))
		x = x.view(x.size()[0], -1)
		x = F.relu(self.h4(x))
		x = F.relu(self.h5(x))
		return x

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)	

class GameAgent():
	def __init__(self, gamma, epsilon, alpha, batchSize, maxBufferSize=50000, minEpsilon=0.0001, decayEpsilon=0.99):
		global eps
		self.epsilon = epsilon
		self.minEpsilon = minEpsilon
		self.decayEpsilon = decayEpsilon
		self.batchSize = batchSize
		self.gamma = gamma
		self.maxBufferSize = maxBufferSize
		# self.memory = deque(maxlen=maxBufferSize)
		# self.bufferItr = 0
		self.brain = Brain(alpha)
		state_dict, self.memory, self.bufferItr, eps = loader("Logs/")
		if state_dict is not None:
			self.brain.load_state_dict(state_dict)
			self.brain.eval()
			self.epsilon = self.minEpsilon
		

	def storeMem(self, state, action, reward, nextState, done):
		if self.bufferItr >= self.maxBufferSize:
			self.bufferItr = self.bufferItr % self.maxBufferSize
			self.memory[self.bufferItr] = [state, action, reward, nextState, done]
			self.bufferItr += 1
		else:
			self.bufferItr += 1
			self.memory.append([state, action, reward, nextState, done])

	def act(self, state):
		if np.random.rand(1) < self.epsilon:
			return np.random.randint(0, 2)
		else:
			return self.brain.forward(state).argmax().item()

	def decay(self):
		if self.epsilon > self.minEpsilon:
			self.epsilon -= (0.1 - 0.0001) / 100000
			# self.epsilon = self.epsilon * self.decayEpsilon
		else:
			self.epsilon = self.minEpsilon

	def train(self):
		if len(self.memory) < 100:
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
		targetQValue = qValue.clone().detach().numpy()
		nextQValue = self.brain.forward(newstates)
		npNextQValue = np.amax(nextQValue.clone().detach().numpy(), axis=1)
		
		actions = actionsTemp.reshape(self.batchSize,).astype(int)
		indexes = np.arange(self.batchSize)
		targetQValue[(indexes, actions)] = rewards * dones + (rewards + npNextQValue * self.gamma) * notdones
		targetQValue = torch.Tensor(targetQValue)

		self.brain.optimizer.zero_grad()
		loss = self.brain.lossFunc(qValue, targetQValue)
		loss.backward()
		self.brain.optimizer.step()
		
myEnv = Dino_env()
dqn = GameAgent(0.99, 0.1, 0.0001, 16)
stat_file = open("stats.txt","a")
counter = 0

myEnv.step(1)
while True:
	st, reward, done_ = myEnv.step(0)
	st = np.stack((st, st, st, st), axis=0)
	currentState = st.reshape(1, st.shape[0], st.shape[1], st.shape[2])
	rewardSum = 0
	curr_score = myEnv.getScore()
	high_score = myEnv.getHighScore()
	done = False

	while not done:
		bestAction = dqn.act(currentState)
		new_st, reward, done = myEnv.step(bestAction)
		new_st = new_st.reshape(1, 1, new_st.shape[0], new_st.shape[1])
		new_state = np.append(new_st, currentState[:, 1:, :, :], axis=1)
		if myEnv.getScore() > curr_score:
			curr_score = myEnv.getScore()

		dqn.storeMem(currentState, bestAction, reward, new_state, done)

		if counter*16 >= 50000:
			myEnv.pause()
			saver(dqn.brain.state_dict(),dqn.memory,dqn.bufferItr, eps, "Logs/", True)
			myEnv.play()
			counter = 0
		else:
			counter += 1

		dqn.train()
		rewardSum += reward
		currentState = new_state
		dqn.decay()

	stat_file.write(str(eps) + ", " + str(curr_score) + "\n")
	print("Reached score: {} in episode: {} with high score: {}, epsilon: {}".format(curr_score, eps, myEnv.getHighScore(), dqn.epsilon))
	eps += 1
	if myEnv.getHighScore() > high_score:
		high_score = myEnv.getHighScore()
		saver(dqn.brain.state_dict(),dqn.memory,dqn.bufferItr, eps, "Logs/best/", True)

myEnv.stop()

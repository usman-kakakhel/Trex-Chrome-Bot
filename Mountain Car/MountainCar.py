import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt


class Brain(nn.Module):
    def __init__(self, inputSize, h1, h2, outputSize, lr):
        super(Brain, self).__init__()
        self.inp = inputSize
        self.out = outputSize
        self.h1 = h1
        self.h2 = h2

        self.hiddenLayer1 = nn.Linear(self.inp, self.h1)
        self.hiddenLayer2 = nn.Linear(self.h1, self.h2)
        self.output = nn.Linear(self.h2, self.out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lossFunc = nn.MSELoss()
    

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.hiddenLayer1(x))
        x = F.relu(self.hiddenLayer2(x))
        return self.output(x)
        
class GameAgent():
    def __init__(self, gamma, epsilon, alpha, batchSize, inputSize, h1Size, 
                h2Size, outputSize, maxBufferSize=100000, minEpsilon=0.01, decayEpsilon=0.995):
        self.epsilon = epsilon
        self.minEpsilon = minEpsilon
        self.decayEpsilon = decayEpsilon
        self.batchSize = batchSize
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.gamma = gamma
        self.maxBufferSize = maxBufferSize
        self.bufferItr = 0
        self.brain = Brain(inputSize, h1Size, h2Size, outputSize, alpha)
        self.memory = deque(maxlen=maxBufferSize)

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




env = gym.make('MountainCar-v0')
dqn = GameAgent(0.995, 1.0, 0.001, 48, env.observation_space.shape[0], 256, 256, env.action_space.n)
for eps in range(400):
    scores=[]
    a=[]
    currentState=env.reset()
    rewardSum = 0
    max_position=-99
    done = False
    i = 0
    while not done:
        env.render()
        bestAction = dqn.act(currentState)
        new_state, reward, done, _ = env.step(bestAction)

        
        if new_state[0] > max_position:
            max_position = new_state[0]

        if new_state[0] >= 0.5:
            reward += 10

        dqn.storeMem(currentState, bestAction, reward, new_state, done)

        dqn.train()
        rewardSum += reward
        currentState = new_state
        i += 1
        dqn.decay()

    if i >= 199:
        print("Failed to finish task in epsoide {}".format(eps))
    else:
        print("Success in epsoide {}, used {} iterations!".format(eps, i))

    print("now epsilon is {}, the reward is {} maxPosition is {}".format( dqn.epsilon, rewardSum,max_position))
    scores.append(max_position)
    a.append(eps)


plt.scatter(a[:], scores[:], s=15, label='passed')
plt.legend()
plt.show()
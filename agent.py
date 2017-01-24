import gym
import time

#import os
#import sys
#sys.path.append(os.getcwd())
import numpy as np
import nn
from csvWrap import *
from filt import *
from nn import printInfo

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

envs=['CartPole-v0','FrozenLake-v0','MountainCar-v0','SpaceInvaders-v0']
i=0
for s in envs:
	print(str(i)+':'+s)
	i+=1
print('choose env:')
try:
	x=int(raw_input())
except:
	print('input error.  setting environment 0.')
	x=0
print('loading '+envs[x])
env_id=envs[x]
env = gym.make(env_id)

task='gym'+str(x)

print('env.action_space:'),
print(env.action_space)
print('env.observation_space:'),
print(env.observation_space)
print('env.reward_range:'),
print(env.reward_range)

INPUTS=env.observation_space.shape[0]
OUTPUTS=env.action_space.n
highVec=env.observation_space.high
lowVec=env.observation_space.low

env.seed(0)
agent = RandomAgent(env.action_space)

episode_count = 100
reward = 0
done = False

RENDER=True
RENDER=False
RENDERTIMESTEP=0.1
VERBOSE=True

BATCHSIZE=10
#raise Exception
#LAYERDIM=[2,1025,2]
#LAYERDIM=[2,500,10,2]
LAYERDIM=[INPUTS,500,OUTPUTS]
EPOCHS=100
GAMMA=0.005
PRINTFREQ=BATCHSIZE

np.set_printoptions(precision=4)

NN=nn.Network(LAYERDIM,GAMMA)
filename=task+'.csv'
try:
	open(filename,'r')
	x=raw_input('found '+filename+'.  load network?  ([y]/n)')
	if(x=='' or x=='y'):
		print('loading '+filename+'...')
		NN=nn.loadNetwork(filename)
except:
	x=raw_input(filename+' not found.  start?  ([y]/n)')
	if(x=='n'):
		exit()
t0=time.clock()

LEARNINGRATE=(0.1)

for episode in range(episode_count):
	obs = env.reset()
	rewardSum=0
	stepSum=0
	bInp=[]
	bOut=[]
	memories=[]
	while True:
		if RENDER:
			env.render()
			time.sleep(RENDERTIMESTEP)
		action = agent.act(obs, reward, done)
		obs, reward, done, _ = env.step(action)
		memories.append(env.step(action)+(action,))
		rewardSum+=reward
		stepSum+=1
		if reward!=0:
			for memory in memories:
				bInp.append(memory[0])
				output=NN.predict(memory[0])
				#output is a numpy array.
				output-=LEARNINGRATE*reward/2.0
				output[memory[4]]+=LEARNINGRATE*reward
				bOut.append(output)
			[output,error]=NN.batchTrain(bInp,bOut)
			#if (epoch%PRINTFREQ==0):
			if True:
				print('----------ep:'+str(episode))
				for i in range(len(bInp)):
					printInfo(error[i],output[i],bOut[i])
		if done:
			if(VERBOSE):
				print('ep:'+str(episode)+',total steps:'+str(stepSum)+',total reward:'+str(rewardSum))
			break

env.close()
tf=time.clock()
print('elapsed time: '+str(tf-t0)+'s')
x=raw_input('save network?  (y/[n])')
if(x=='y'):
	NN.save(filename)

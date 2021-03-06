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

class agent(object):
	def __init__(self,env):
		self.action_space=env.action_space
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
		#LAYERDIM=[2,1025,2]
		#LAYERDIM=[2,500,10,2]
		LAYERDIM=[INPUTS,500,OUTPUTS]
		GAMMA=0.005

		self.NN=nn.Network(LAYERDIM,GAMMA)
		filename=task+'.csv'
		self.filename=filename
		try:
			open(filename,'r')
			x=raw_input('found '+filename+'.  load network?  ([y]/n)')
			if(x=='' or x=='y'):
				print('loading '+filename+'...')
				self.NN=nn.loadNetwork(filename)
		except:
			x=raw_input(filename+' not found.  start?  ([y]/n)')
			if(x=='n'):
				exit()
	def random(self):
		return self.action_space.sample()
	def act(self,obs):
		output=self.NN.predict(obs)
		action=np.argmax(output)
		return action
	def save(self):
		self.NN.save(self.filename)
	def learn(self,memories,reward):
		futureRew=reward
		for i in range(len(memories)-1,-1,-1):
			mem=memories[i]
			act=mem[4]
			rew=mem[1]
			ob=mem[0]
			output=self.NN.predict(ob)
			change=(1-ALPHA)*output[act]+ALPHA*(rew+GAMMA*futureRew)
			if np.fabs(change-output[act])>0.1:
				bInp.append(ob)
				output[act]+=change
				bOut.append(output)
			futureRew=np.argmax(output)
			#if(np.argmax(output)>0.99):
			#	print((output))
			#if(np.argmin(output)<-0.99):
			#	print((output))
		[output,error]=self.NN.batchTrain(bInp,bOut)
#			meanError=4
#			while meanError>1:
#				[output,error]=NN.batchTrain(bInp,bOut)
#				meanError=np.mean(np.fabs(error))
			#print('meanError:'+str(meanError))
		if False:
			print('----------ep:'+str(episode))
			for i in range(len(bInp)):
				printInfo(error[i],output[i],bOut[i])


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

env.seed(0)
#agent = RandomAgent(env.action_space)
agent = agent(env)

episode_count = 10
reward = 0
done = False

RENDER=True
RENDER=False
RENDERTIMESTEP=0.1
VERBOSE=True

np.set_printoptions(precision=4)
t0=time.clock()
ALPHA=0.2
GAMMA=0.5
LEARNINGRATE=(0.1)
info=0
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
		if(episode<5):
			action=agent.random()
		else:
			action=agent.act(obs)
		memory=(obs,reward,done,info,action)
		#print('action='+str(action))
		results = env.step(action)
		obs,reward,done,info=results
		memories.append(memory)
		rewardSum+=reward
		stepSum+=1
		if done:
			agent.learn(memories,reward)
			if(VERBOSE):
				print('ep:'+str(episode)+',total steps:'+str(stepSum)+',total reward:'+str(rewardSum))
			break

env.close()
tf=time.clock()
print('elapsed time: '+str(tf-t0)+'s')
x=raw_input('save network?  (y/[n])')
if(x=='y'):
	agent.save()

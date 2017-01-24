import gym
import time

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
	x=0
env_id=envs[x]
env = gym.make(env_id)

print('env.action_space:'),
print(env.action_space)
print('env.observation_space:'),
print(env.observation_space)
print('env.reward_range:'),
print(env.reward_range)

env.seed(0)
agent = RandomAgent(env.action_space)

episode_count = 100
reward = 0
done = False

RENDER=True
RENDER=False
RENDERTIMESTEP=0.1
VERBOSE=True

for i in range(episode_count):
	obs = env.reset()
	rewardSum=0
	stepSum=0
	while True:
		if RENDER:
			env.render()
			time.sleep(RENDERTIMESTEP)
		action = agent.act(obs, reward, done)
		obs, reward, done, _ = env.step(action)
		rewardSum+=reward
		stepSum+=1
		if done:
			if(VERBOSE):
				print('ep:'+str(i)+',total steps:'+str(stepSum)+',total reward:'+str(rewardSum))
			break

env.close()

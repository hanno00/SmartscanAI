import gym
from stable_baselines import ppo1
from stable_baselines.common.policies import MlpPolicy
import numpy as np
import time

from learning_env.foot_env import FootEnv
env = FootEnv()
env.reset()

training = True
continueTraining = False
iters = 3
save_file = ""

if training:
    if continueTraining:
        print("Continue training")
        model = ppo1.load(save_file,env=env,tensorboard_log="tensorboard_logs/")
    else:
        print("Train new model")
        model = ppo1(MlpPolicy, env=env, verbose=1, tensorboard_log="tensorboard_logs/")
    print("Start learning")     
    model.learn(total_timesteps=iters)
    print("Done learning")
    model.save(save_file) 
else:
    model = ppo1.load(save_file,env=env)

obs = env.reset()
print("reseting env")
done = False
hists = []
for i in range(10):
    obs = env.reset()
    hist = []
    while not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        hist.append(reward)
    som = np.sum(hist)
    hists.append(som)
    print("end reward", reward,"\ttotal reward",som)
    print(obs)
    done = False


env.close()

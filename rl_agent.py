# disable unimportant warnings
import warnings
import os

from numpy.lib.npyio import save
warnings.filterwarnings("ignore",category=DeprecationWarning,message='.*') 
warnings.filterwarnings("ignore",category=FutureWarning,message='.*')
warnings.filterwarnings("ignore",category=PendingDeprecationWarning,message='.*')

# import RL modules
from stable_baselines3 import PPO
import numpy as np

# import custom classes
from FootEnvironment import FootEnv
from Augmentation import Augmentation

class Agent():
    @staticmethod
    def training(pc_folder,save_file,iters,training,continueTraining,steps_max,size_model,prints=False):
        # init env
        env = FootEnv(pc_folder,prints,max_steps=steps_max,model_size=size_model)
        env.reset()

        # train model
        if training:
            if continueTraining:
                print("Continue training")
                model = PPO.load(save_file,env=env,tensorboard_log="trained_models/tensorboard_logs/")
            else:
                print("Train new model")
                model = PPO('MlpPolicy', env=env, verbose=1, tensorboard_log="trained_models/tensorboard_logs/")
            print("Start learning")     
            model.learn(total_timesteps=iters)
            print("Done learning")
            model.save(save_file) 
        else:
            model = PPO.load(save_file,env=env)

        # test model
        obs = env.reset()
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

    @staticmethod
    def predict(save_file,pc,pc_folder,path):
        # init env
        env = FootEnv(pc_folder,prints=False)
        obs = env.reset(pc)
        model = PPO.load(save_file,env=env)
        done = False
        while not done:
                action, state = model.predict(obs)
                obs, reward, done, info = env.step(action)
        np.savetxt(path+'.csv', env.pc, delimiter=',',header='X,Y,Z',comments='')
        return env.pc

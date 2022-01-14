# import RL modules
from stable_baselines3 import PPO
import numpy as np

# import custom classes
from FootEnvironment import FootEnv

class Agent():
    @staticmethod
    def training(pc_folder,save_file,iters,continueTraining,steps_max,size_model,prints=False):
        # init env
        env = FootEnv(pc_folder,prints,max_steps=steps_max,model_size=size_model)
        env.reset()

        # train model
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
        env.close()

    @staticmethod
    def predict(save_file,pc,pc_folder,path,prints=False,max_steps=20,model_size=1000):
        # init env
        env = FootEnv(pc_folder,prints,max_steps=max_steps,model_size=model_size)
        env = FootEnv(pc_folder,prints=False)
        obs = env.reset(pc)
        model = PPO.load(save_file,env=env)
        done = False
        while not done:
                action, state = model.predict(obs)
                obs, reward, done, info = env.step(action)
        np.savetxt(path+'.csv', env.pc, delimiter=',',header='X,Y,Z',comments='')
        output_pc = env.pc
        env.close()
        return output_pc

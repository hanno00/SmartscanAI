# import RL modules
from stable_baselines3 import PPO
import numpy as np

# import custom classes
from FootEnvironment import FootEnv
from Augmentation import Augmentation
from Preprocessing import Preprocessing

# settings
generate_new_clouds = False
training = True
continueTraining = True
iters = 1
save_file = "trained_models/PPO/testing"
pc_folder = "ply_out"

# regenerate dataset if needed
if generate_new_clouds:
    Augmentation.augment_folder("original_point_clouds",pc_folder,csv=False)

# init env
env = FootEnv(pc_folder,prints=True)
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
for i in range(1):
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

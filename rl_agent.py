# disable unimportant warnings
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning,message='.*') 
warnings.filterwarnings("ignore",category=FutureWarning,message='.*')
warnings.filterwarnings("ignore",category=PendingDeprecationWarning,message='.*')

# import RL modules
from stable_baselines import PPO1
from stable_baselines.common.policies import MlpPolicy
import numpy as np

# import custom classes
from FootEnvironment import FootEnv
from Augmentation import Augmentation

# settings
generate_new_clouds = False
training = True
continueTraining = False
iters = 3
save_file = "trained_models/PPO1/testing"
pc_folder = "pc_out"

# regenerate dataset if needed
if generate_new_clouds:
    Augmentation.augment_folder("original_point_clouds",pc_folder)

# init env
env = FootEnv(pc_folder,prints=True)
env.reset()

# train model
if training:
    if continueTraining:
        print("Continue training")
        model = PPO1.load(save_file,env=env,tensorboard_log="trained_models/tensorboard_logs/")
    else:
        print("Train new model")
        model = PPO1(MlpPolicy, env=env, verbose=1, tensorboard_log="trained_models/tensorboard_logs/")
    print("Start learning")     
    model.learn(total_timesteps=iters)
    print("Done learning")
    model.save(save_file) 
else:
    model = PPO1.load(save_file,env=env)

# test model
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

import open3d as o3d
import numpy as np
import gym
from gym import spaces


from PcdController import PcdController as cloud
from Augmentation import Augmentation

class FootEnv(gym.Env):    
    def __init__(self,pointclouds_location,prints=False,max_steps=200,model_size=3500):
        # set constants
        self.PRINT = prints
        self.MAX_STEPS = max_steps
        self.MODEL_SIZE = model_size
        self.action_space = spaces.Box(low=10,high=10,shape=(model_size*3,),dtype=float) # max 1 cm verplaating per punt
        self.observation_space = spaces.Box(low=-500,high=500,shape=(model_size*3,),dtype=float) # min and max 50cm voor de voet, in alle richtingen
        self.CLOUDS = Augmentation.load_folder_ply(pointclouds_location)
        self.N_CLOUDS = len(self.CLOUDS)
        print(f"Loaded {self.N_CLOUDS} point clouds into memory")
        print(f'Epochs will termintate after {self.MAX_STEPS} timestaps')

        # intit variables
        self.__change_cloud() #inits both self.pcd and self.pc
        self.scores = []
        self.distances = []
        self.timestep = 0
        return None

    def reset(self):
        self.printt("Resetting environment...")
        self.timestep = 0
        self.scores = []
        self.distances = []
        self.__change_cloud()
        return self.pc.flatten()
    
    def step(self, actions):
        done = False
        self.timestep += 1

        # convert action vector to a action matrix
        actions = actions.reshape(self.MODEL_SIZE,3)
        # make new pointcloud and update pcd
        self.pc = np.add(self.pc,actions)
        self.pcd.points = o3d.utility.Vector3dVector(self.pc)

        # update score
        score = self.__calc_score()
        self.scores.append(score)

        # update distances 
        distance = self.__calc_moved_distance(actions)
        self.distances.append(distance)
 
        # reward
        reward = 200 + score * -50 + distance * -0.01

        # check if done
        if self.timestep > self.MAX_STEPS:
            done = True
        
        # update info
        info = {
            "timestep":self.timestep,
            "score":score,
            "distance_moved":distance,  
        }
        self.printt(f"Timestep {self.timestep}: reward = {reward}")

        return self.pc.flatten(), reward, done, info

    def render(self,title="PointCloud"):
        cloud.draw_pcd(self.pcd,title)

    def __change_cloud(self):
        self.printt("Cloud changed to random cloud")
        rand = np.random.randint(0,self.N_CLOUDS)
        self.pcd = self.CLOUDS[rand]
        self.pc = self.pcd_to_array(self.pcd)
        assert len(self.pc)==self.MODEL_SIZE,"Loaded point cloud does not match the expected size!"

    def __calc_score(self):
        pcd = cloud.computing_normals(self.pcd,20,5)
        pcd = cloud.orient_normals(pcd,10)
        score = cloud.compute_cost(pcd,5)
        self.printt(f"Model smoothness score: {score}")
        return score      

    def pcd_to_array(self,pcd):
        return np.asarray(pcd.points)
    
    def printt(self,str=''):
        if self.PRINT:
            print(str)

    def __calc_moved_distance(self,actions):
        squares = np.square(actions)
        total = 0
        for row in squares:
            total += np.sqrt(sum(row))
        self.printt(f'Distance moved: {total}')
        return total




import open3d as o3d
import numpy as np
import gym
from gym import spaces


from PcdController import PcdController as cloud
from Augmentation import Augmentation

class FootEnv(gym.Env):    
    def __init__(self,pointclouds_location,prints=False,max_steps=200,model_size=3000):
        # set constants
        self.PRINT = prints
        self.MAX_STEPS = max_steps
        self.MODEL_SIZE = model_size
        self.action_space = spaces.Box(low=20,high=20,shape=(model_size,3),dtype=float) # max 2 cm verplaating per punt
        self.observation_space = spaces.Box(low=-500,high=500,shape=(model_size,3),dtype=float) # min and max 50cm voor de voet, in alle richtingen
        self.CLOUDS = Augmentation.load_folder(pointclouds_location)
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
        self.printt("Reset called")
        self.timestep = 0
        self.scores = []
        self.distances = []
        self.n_points_deleted = []
        self.__change_cloud()
        return self.pc
    
    def step(self, actions):
        done = False
        self.timestep += 1

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
        reward = 500 + score * -50 + distance * -200

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

        return self.pc, reward, done, info

    def render(self,title="PointCloud"):
        cloud.draw_pcd(self.pcd,title)

    def __change_cloud(self):
        self.printt("Cloud change called")
        rand = np.random.randint(0,self.N_CLOUDS)
        pcd = self.CLOUDS[rand]

        length = len(pcd.points)
        if length > self.MODEL_SIZE:
            self.pcd = cloud.down_sample(pcd,self.MODEL_SIZE)
            self.printt(f'Model of size {length} down sampled to size {self.MODEL_SIZE}')
        elif length < self.MODEL_SIZE:
            self.pcd = Augmentation.upsample(pcd,20,self.MODEL_SIZE)
            self.printt(f'Model of size {length} up sampled to size {self.MODEL_SIZE}')
        
        self.pc = self.pcd_to_array(self.pcd)

    def __calc_score(self,pcd):
        pcd = cloud.computing_normals(pcd,20,5)
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




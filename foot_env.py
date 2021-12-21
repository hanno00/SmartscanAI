import math
import open3d as o3d
import numpy as np
import gym
from gym import spaces


import Triangulation as triangles
from Augmentation import Augmentation

class FootEnv(gym.Env):
    def __str__(self) -> str:
        return super().__str__()
    
    def __init__(self,pointclouds_location,max_perc_del=0.1,max_steps=200):
        self.clouds = Augmentation.load_folder(pointclouds_location)
        self.n_clouds = len(self.clouds)
        print(f"Loaded {self.n_clouds} point clouds into memory")
        self.pcd = self.clouds[0]
        self.pc = self.pcd_to_array(self.pcd)
        self.points_in_cloud = self.pc.shape[0]
        self.scores = []
        self.n_points_deleted = []

        self.MAX_STEPS = max_steps # maximale timesteps
        self.MAX_DELETED = max_perc_del # float range 0-1, 1=max 100% verwijderen
        self.timestep = 0
        self.__set_space_size(self.points_in_cloud)
        return None

    def reset(self):
        self.timestep = 0
        self.scores = []
        self.n_points_deleted = []
        self.__change_cloud()
        return self.pc
    
    def step(self, actions):
        done = False
        self.timestep += 1

        # delete points
        points_deleted = sum(actions)
        self.n_points_deleted.append(points_deleted)
        n_points = self.__delete_points(actions)

        # update score
        score = self._calc_score()
        self.scores.append(score)

        # reward 
        reward = 5
        reward -= points_deleted * -10
        reward += score * 100
 
        # check if done
        percent_deleted = n_points / self.points_in_cloud 
        if self.timestep > self.MAX_STEPS or percent_deleted > self.MAX_DELETED:
            done = True
        
        # update info
        info = {
            "timestep":self.timestep,
            "score":score,
            "points":n_points,
            "initial_points":self.points_in_cloud,
            "percent_deleted":percent_deleted,    
        }

        return self.pc, reward, done, info

    def render(self,option):
        o3d.vi
        pass

    def __change_cloud(self):
        rand = np.random.randint(0,self.n_clouds)
        self.pcd = self.clouds[rand]
        self.pc = self.pcd_to_array(self.pcd)
        self.points_in_cloud = self.pc.shape[0]

    def _calc_score(self):
        normals = triangles.getNormals()
        angles = triangles.getAngles(normals)
        return max(angles)

    def __delete_points(self,actions):
        self.pc = self.pc[actions] # omdat action = [0,1] kan de array gebruikt worden als mask
        points_left = self.pc.shape[0] #might need to be index 1 or 2 or 3 or something, idk how the shape var looks like
        self.__set_space_size(points_left)
        return points_left

    def __set_space_size(self,size):
        self.action_space = spaces.Box(low=0,high=1,shape=(size,1)) # 0 = delete, 1 = no-delete
        self.observation_space = spaces.Box(low=-500,high=500,shape=(size,3)) # min and max 50cm voor de voet, in alle richtingen

    def pcd_to_array(pcd):
        return np.asarray(pcd.points)




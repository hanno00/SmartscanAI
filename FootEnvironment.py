import open3d as o3d
import numpy as np
import gym
from gym import spaces


from PcdController import PcdController as cloud
from Augmentation import Augmentation

class FootEnv(gym.Env):    
    def __init__(self,pointclouds_location,prints=False,max_perc_del=0.1,max_steps=200,model_size=3000):
        self.print = prints
        self.clouds = Augmentation.load_folder(pointclouds_location)
        self.n_clouds = len(self.clouds)
        print(f"Loaded {self.n_clouds} point clouds into memory")
        self.pcd = self.clouds[0]
        self.pc = self.pcd_to_array(self.pcd)
        self.points_in_cloud = self.pc.shape[0]
        self.__set_space_size(self.points_in_cloud)
        self.scores = []
        self.n_points_deleted = []

        self.MAX_STEPS = max_steps # maximale timesteps
        self.MAX_DELETED = max_perc_del # float range 0-1, 1=max 100% verwijderen
        self.MODEL_SIZE = model_size
        self.timestep = 0
        print(f'Epochs will termintate after {self.MAX_DELETED:.0%} points deleted or {self.MAX_STEPS} timestaps')
        return None

    def reset(self):
        self.printt("Reset called")
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
        score = self.__calc_score()
        self.scores.append(score)

        # reward 
        reward = 100
        reward -= points_deleted * 10
        reward -= score * 100
 
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
        self.printt("")

        return self.pc, reward, done, info

    def render(self,title="PointCloud"):
        cloud.draw_pcd(self.pcd,title)

    def __change_cloud(self):
        self.printt("Cloud change called")
        #rand = np.random.randint(0,self.n_clouds)
        rand = 0
        self.pcd = self.clouds[rand]
        self.pc = self.pcd_to_array(self.pcd)
        self.points_in_cloud = self.pc.shape[0]
        self.printt(f"Shape of new cloud {self.pc.shape}")
        self.__set_space_size(self.points_in_cloud)

    def __calc_score(self):
        pcd = cloud.computing_normals(self.pcd,20,5)
        pcd = cloud.orient_normals(pcd,10)
        return cloud.compute_cost(pcd,5)
        

    def __delete_points(self,actions):
        actions = actions > 0 #convert float action to boolean actions
        self.pc = self.pc[actions] # omdat action = [0,1] kan de array gebruikt worden als mask
        #self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.pc)
        points_left = self.pc.shape[0]
        self.printt(f"Deleted some points, {points_left} points remaining")
        self.__set_space_size(points_left)
        return points_left


    def __set_space_size(self,size):
        self.printt(f"Set size of observation space to ({size}, 3)")
        self.action_space = spaces.MultiBinary(size) # 0 = delete, 1 = no-delete
        self.observation_space = spaces.Box(low=-500,high=500,shape=(size,3),dtype=float) # min and max 50cm voor de voet, in alle richtingen

    def pcd_to_array(self,pcd):
        return np.asarray(pcd.points)
    
    def printt(self,str):
        if self.print:
            print(str)




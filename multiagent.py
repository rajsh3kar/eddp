import numpy as np

import gymnasium
import sys
from itertools import cycle
import functools
from comm import *
import random
import matplotlib 
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from user_mobility import RWPM
from matplotlib.animation import FuncAnimation
from gaussmobility import GaussMobilityModel
from matplotlib import style
matplotlib.use( 'tkagg' )
writer = SummaryWriter()
style.use("ggplot")
# for ploting the animated graph
plt.ion()
fig, ax = plt.subplots(1, figsize=(4, 4))
plt.tight_layout()
alpha = 0.5
max_step = 1000
class UE():
    def __init__(self,idx):
        self.idx = idx
        
        self.assigned = 0
        self.assigned_dbs = 0
        self.sinr= {}
        self.mobility_model = GaussMobilityModel(AREA,1000,1)
        self._position= self.mobility_model.user_position
    
    def get_position(self):
        
        return self.mobility_model.user_position
    def set_position(self):
        self.mobility_model.initialize_user_position()
        self._position= self.mobility_model.user_position
        self.posi_gen=self.mobility_model.simulate_mobility()
    def update_position_ue(self):
         self._position = cycle(self.posi_gen)
    def set_assigned(self,assigned):
        self.assigned = assigned
        
    def is_assigned(self):
        return  self.assigned

TRANSMISSION_ANGLE = math.pi/3


class DDP_multi_env(gymnasium.Env):

    def __init__(self,n_ue):
        #random.seed(random_seed)
        self.observation_space =gymnasium.spaces.Box(low=np.array(
            [[0, 0, A_min, 0,]]*N_DBS ), high=np.array([[AREA, AREA ,A_max, B]]*N_DBS, dtype=np.int32))
        self.action_space = gymnasium.spaces.Box(low=np.array([-1]*N_DBS), high=np.array([1]*N_DBS),dtype = np.int32)
        
        self.n_ue = n_ue
        self.drone_postion = None
        
        self.state = np.array([0,0,0,0]*(N_DBS))
        self.assigned_ue = []
        self.n_connected_ues = 0
        self._UEs = [UE(j+1)for j in range(self.n_ue)]
        
        self.is_done = False
        self.mobility_model = RWPM((0,AREA),(0,AREA),5,10, 10)
        self.prev_connected = 0
    def set_dbs_position(self,dbs_position):
        self.dbs_position = dbs_position
    
    
    def get_dbs_position(self):
        return self.dbs_position
    
    #check whther the db is out of range
    def is_db_out_of_range(self,i):
           #print(self.dbs_position[i][0] >= AREA or self.dbs_position[i][0] <= 0 or self.dbs_position[i][1] >= AREA or self.dbs_position[i][0] <= 0)
           return  self.dbs_position[i][0] >= AREA or self.dbs_position[i][0] <= 0 or self.dbs_position[i][1] >= AREA or self.dbs_position[i][0] <= 0
    #Update the dbs position
    def update_db_position(self,action):
        
        for i in range(N_DBS):
            self.dbs_position[i][0] = self.dbs_position[i][0] + unit_move * math.cos(action[i])
            self.dbs_position[i][1] = self.dbs_position[i][1] + unit_move * math.sin(action[i])
            self.dbs_position[i] = np.clip(self.dbs_position[i],0,AREA)

    

    def update_bandwidth(self,i,bandwidth):

        self._bandwidth[i] += bandwidth       
            
    def transmission_radius(self,i,distance):
        return self._altitude[i] * math.tan(TRANSMISSION_ANGLE)
            

            
    
    def set_altitude(self,alti):
        self._altitude = alti
    
    def get_altitude(self):
        return self._altitude
    
    def find_nearby(self,i):
        key = self.dbs_position[i]
        distances = np.linalg.norm(self.dbs_position-key,axis =1)
        distances[distances == 0] = 1000000000000
        nearby_db_index = np.nonzero(distances<30)
        nearby_drones = self.dbs_position[nearby_db_index]
        return nearby_db_index
   
    
    def assign_ue_to_dbs(self):
        total_assinged = 0
        self.n_connected_ues = [0 for _ in range(N_DBS)]
        for ue in self._UEs:
            ue.assigned = 0
        self._bandwidth = [0 for _ in range(N_DBS)]
        for i in range(N_DBS):
            for ue in self._UEs:
                I=0
                distance = self.distance_dbs_ue(i,self.dbs_position[i],ue)
                nearby = self.find_nearby(i)
                #print(f'nearby drone for {i} is {nearby}')
                if distance <= self.transmission_radius(i,distance):
                        sinr = SINR(self._altitude[i],distance)
                       
                        if(sinr >= sinr_th):
                            
                            if(ue.is_assigned() == 0 and self._bandwidth[i] <= B):
                                self.n_connected_ues[i] += 1
                                ue.set_assigned(1)
                                #print(f"DBS {dbs.idx}: connected UEs {dbs.n_connected_ues}")
                                b = bandwidth(self._altitude[i],distance)
                                self.update_bandwidth(i,b)
                            

                                total_assinged += 1
                                    
                                
                            else:
                                pass
      
    


    def distance_dbs_ue(self,i,dbs,ue):
        temp = int(np.linalg.norm(dbs - ue.get_position()))
        #print(temp)
        distance =  math.sqrt(temp**2 + self._altitude[i]**2)
        return distance

    
    def get_reward(self,i):   
    # reward design in casesbased on the observed connected ues    
        
        if(self.n_connected_ues[i] == self.prev_connected[i] and self.n_connected_ues[i] != 0):
            if(self._bandwidth[i] <= self.prev_bandwidth[i]):
              
                
                self.rewards[i] = self.rewards[i]+ 1/self._bandwidth[i]
                #print(self.rewards[i])
            else:
                self.rewards[i] = self.rewards[i]- 1/self.prev_bandwidth[i]
                
        elif(self.n_connected_ues[i] > self.prev_connected[i]):
            self.rewards[i] = self.rewards[i] + (self.n_connected_ues[i] - self.prev_connected[i])
        
        elif(self.n_connected_ues[i] < self.prev_connected[i]):
            self.rewards[i] = self.rewards[i] - (self.prev_connected[i]-self.n_connected_ues[i])

        
        return self.rewards[i]
            
           
      
        
        
       

   
    
    def reset(self,seed = None, options=None):
      
       #print(self.agents)
       
       #self.observation = self.observation_space.sample()
       #self.dbs_position = 
       self.dbs_position = np.array([np.random.uniform(0, AREA, size=(2,)) for _ in range(N_DBS)],dtype=np.int32)
       self._altitude = np.array([random.randint(50,150) for _ in range(N_DBS)])
       self._bandwidth = np.array([0.0 for _ in range(N_DBS)])
       self.n_connected_ues = np.array([0 for _ in range(N_DBS)])
       self.sinr = np.array([0.0 for _ in range(N_DBS)])
       self.n_prev_connected_ues = np.array([0 for _ in range(N_DBS)])
       self.prev_bandwidth = np.array([0.0 for _ in range(N_DBS)])
       self.state = np.column_stack((self.dbs_position,self._altitude,self._bandwidth))
       self.rewards = np.array([0 for _ in range(N_DBS)],dtype=np.float32)
             
       for ue in self._UEs:

            ue.set_position()
       self.observation = self.state
       
       return self.state, {}
    
    def get_state(self):
        return np.column_stack((self.dbs_position,self._altitude,self._bandwidth))

    def action_normalize(self,actions):
        act = 180*(1+actions)
        return act
    def step(self, action):
       
        actions = self.action_normalize(action)
        
        #self.rewards = np.array([0 for _ in range(N_DBS)])
        done = False
        time_step = 0
        #print('actions are ',actions)
        self.prev_connected = self.n_connected_ues
        self.prev_bandwidth = self._bandwidth
        
       
        self.update_db_position(actions)
                    
        self.assign_ue_to_dbs()
        

        #store back the updated state
        self.state = self.get_state()
       #print(self._bandwidth)
        self.term = False
        for i in range(N_DBS):
            terminated = bool(np.size(self.find_nearby(i)))
            self.term = terminated
            if terminated or self.is_db_out_of_range(i) :
                
                #self.rewards[i] = -10000
                break
                          
            else:
                #print(i)
                self.rewards[i] = self.get_reward(i)
    

       
        time_step += 1

        #update ue position according to gauss mobility 
        for i in range(self.n_ue):
            self._UEs[i].update_position_ue()
       
        reward = sum(self.rewards)
        
        #print(self.observation.shape)
        return self.state,reward,self.term,False,{}


            
        



    def render(self):
        plt.clf()
        
        
        temp_x_dbs =[]
        temp_y_dbs = []
        temp_x_ue = []
        temp_y_ue = []
        for dbs in range(N_DBS):
            temp_x_dbs .append(self.dbs_position[dbs][0])
            temp_y_dbs.append(self.dbs_position[dbs][1])
        for ue in self._UEs:
            temp_x_ue.append(ue.get_position()[0])
            temp_y_ue.append(ue.get_position()[1])
        plt.plot(temp_x_dbs,temp_y_dbs,'rX')
        plt.plot(temp_x_ue,temp_y_ue,'bo')

        plt.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
'''
env = DDP_multi_env(100)
env.reset()

print(env.step(env.action_space.sample()))

import time
while True:
   
    env.step(env.action_space.sample())
    print('Total Rewards: ',np.sum(env.rewards))
    print('Total Connected UE and Bandwidth utilised',sum(env.n_connected_ues),np.divide(env._bandwidth,B) )
    env.render()
    #time.sleep(1)'''

'''env = DDP_multi_env(100)
from gymnasium.envs.registration import register
# Example for the CartPole environment
register(
    # unique identifier for the env `name-version`
    id="eddp:eddp",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=env
   
)
           
env= gymnasium.make('eddp:eddp')   



'''





        



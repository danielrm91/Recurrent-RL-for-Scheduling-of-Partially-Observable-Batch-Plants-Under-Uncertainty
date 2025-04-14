import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class ProcessEnv(gym.Env):           #action    #accumulated times
    def __init__(self):   
        self.nUnits = 4
        self.nProducts = 4
        self.nActions = 7
        self.max_steps = 50
        self.batches = [8,10,5,8,5,8]
        self.warehouse =  [8,8,8,8]
        self.allunits = 7
        self.recipes = [[1,1,0,0,0,1,1],[1,0,1,0,0,1,1],[0,1,0,1,0,0,1],
                        [1,0,0,0,1,0,1],[1,0,1,1,0,0,1],[1,1,0,0,0,0,1]]

        self.recipes_i = [[0,1,5,6],[0,2,5,6],[1,3,6],[0,4,6],[0,2,3,6],[0,1,6]]
        
        self.observation_space = spaces.Dict({"agent" : spaces.Box(0, 1, shape=(2,), dtype=int)})
        self.action_space = spaces.Discrete(6)
    
    def step1(self, action, obs, intimes,times):
        reward = 0
        penalty = 0
        ol = 0
        w = 0
        done = False
        #print('here',obs)    
        #print('intimes',intimes)
        # A grow +1 at every step or -1 if process done
        for i in range(len(intimes)*(self.allunits + 1)):
            if obs[i] > 0:                    # +1 if its a machine that is being occupied
                obs[i] += 1              
            if obs[i] < 0:                    # If it's a -1, then stay like that
                obs[i] = -1
        for j in range(self.allunits,len(intimes)*(self.allunits+1),(self.allunits+1)):     
            if obs[j] != 0:                   # adjusting the actions that are augmented in one
                obs[j] -= 1  
                
        #if done with that unit, go to the next one
        for i in range(len(intimes)): 
            if intimes[i] != 0:
                for j in range(self.allunits):       
                    index = i * (self.allunits+1) + j
                    if obs[index] > times[intimes[i]-1][j]:
                        obs[index] = -1                        
                        ones = sum(1 for i in obs[i*(self.allunits+1):(i+1)*(self.allunits+1)] if i < 0)
                        if ones != 0 and ones < len(self.recipes_i[intimes[i]-1]):
                            ix = i * (self.allunits+1) + self.recipes_i[intimes[i]-1][ones]
                            obs[ix] += 1
                        if ones == len(self.recipes_i[intimes[i]-1]): #if all  are done then erase the -1s 
                            obs[i*(self.allunits+1):(i+1)*(self.allunits+1)] = [0,0,0,0,0,0,0,0] 
                                    
        # Start in unit 1
        if action != 0:   
            index = -(self.allunits + 1 + self.nProducts + 1)    #units + action + products + time (20+1+10+1)
            obs[index + self.recipes_i[action-1][0]] += 1        #start the first machine of that action
            obs[index + self.allunits] = action                  #add action at the end
            intimes[-1] = action                                 #register the action in intimes   
            #create vectors "busy from" and "busy until"
            busy = [[],[]]
            for i in range (len(intimes)):
                if i != 0 and intimes[i] != 0:                    #For an action at time > 0
                    a = intimes[i] - 1 
                    vec = np.array(self.recipes[a]) * i
                    busy[0].append(vec)
                    busy[1].append(vec.copy())
                    for j in range(self.allunits-1):                     
                        if busy[0][-1][j+1] != 0:
                            busy[0][-1][j+1] += sum(times[a][0:j+1])
                    for k in range(self.allunits):
                        if busy[1][-1][k] != 0:
                            busy[1][-1][k] += sum(times[a][0:k+1])
                if i == 0 and intimes[i] !=0:                       #For an action at time 0
                    a = intimes[i] - 1 
                    vec = np.array(self.recipes[a])
                    busy[0].append(vec)
                    busy[1].append(vec.copy())
                    for j in range(self.allunits):
                        if busy[0][-1][j] != 0:
                            busy[0][-1][j] += sum(times[a][0:j]) -1
                    for k in range(self.allunits):
                        if busy[1][-1][k] != 0:
                            busy[1][-1][k] += sum(times[a][0:k+1]) -1                                  
        #PENALTY for current or future overlapping
        if action != 0:
            #############################################
            #if the action exceeds the time limit    
            for i in range(self.allunits):   
                if busy[0][-1][i] > self.max_steps:
                    penalty -= 200
                if busy[1][-1][i] > self.max_steps:
                    penalty -= 200
                    #action = 0
            #############################################        
            #overlapping at other units in the future
            no_act = len(intimes) - np.count_nonzero(intimes==0) - 1     #actions already in the busy vectors
            for i in range (no_act):  
                u = 0 
                #they will be compared with the new one (-1)
                for j in range (self.allunits):
                    if busy[0][i][j] <= busy[0][-1][j] < busy[1][i][j] and busy[0][-1][j]-busy[1][-1][j]!=0:
                        penalty -= 8
                        ol += 1
                        u += 1
                    elif busy[0][i][j] < busy[1][-1][j] <= busy[1][i][j] and busy[0][-1][j]-busy[1][-1][j]!=0: 
                        penalty -= 8   
                        ol += 1
                        u += 1
                    elif busy[0][-1][j] <= busy[0][i][j] < busy[1][-1][j] and busy[0][i][j]-busy[1][i][j]!=0:
                        penalty -= 8
                        ol += 1
                        u += 1
                    elif busy[0][-1][j] < busy[1][i][j] <= busy[1][-1][j] and busy[0][i][j]-busy[1][i][j]!=0:
                        penalty -= 8
                        ol += 1
                        u += 1                        
                    else:
                        reward = 5
                if u != 0:
                    w += 1
                        
            if action == 1 or action == 2:             #Production of P1 
                obs[-5] -= self.batches[action - 1]    #substract the batch
                if obs[-5] > 0:
                    reward += 8
                if -self.warehouse[0] < obs[-5] < 0:
                    reward += 0
                if -self.warehouse[0] > obs[-5]:
                    penalty -= 3
                    obs[-5] += self.batches[action - 1]  #if less than warehouse then restablish the demand

            if action == 3 or action == 4:            #Production of P2 
                obs[-4] -= self.batches[action-1]
                if obs[-4] > 0:
                    reward += 8
                if -self.warehouse[1] < obs[-4] < 0:
                    reward += 0
                if -self.warehouse[1] > obs[-4]:
                    penalty -= 3
                    obs[-4] += self.batches[action - 1]  #if less than warehouse then restablish the demand
                    
            if action == 5:            #Production of P2 
                obs[-3] -= self.batches[action-1]
                if obs[-3] > 0:
                    reward += 8
                if -self.warehouse[2] < obs[-3] < 0:
                    reward += 0
                if -self.warehouse[2] > obs[-3]:
                    penalty -= 3
                    obs[-3] += self.batches[action - 1]    
                    
            if action == 6:            #Production of P2 
                obs[-2] -= self.batches[action-1]
                if obs[-2] > 0:
                    reward += 8
                if -self.warehouse[3] < obs[-2] < 0:
                    reward += 0
                if -self.warehouse[3] > obs[-2]:
                    penalty -= 3
                    obs[-2] += self.batches[action - 1] 
                    
            
        #if there is a penalty, supreme the reward            
        if penalty != 0:
            reward = 0
        
        # finish episode if max steps achieved
        obs[-1] += 1    
        if obs[-1] == self.max_steps:
            done = True
            
        #reward of +1 to minimize time steps
        for i in range(self.allunits,len(intimes)*(self.allunits+1),(self.allunits+1)):
            if obs[i] != 0:
                reward += 0.2
                
        # Add up rewards and penalties
        #print('new obs', obs)
        reward = reward + penalty
        
        #NEW STATE
        return obs, reward, done, intimes, ol, w   
    
    def reset(self):
        pass
    
    def render (self):
        pass
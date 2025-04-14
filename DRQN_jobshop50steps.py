import random
import math
import gym
import csv
import gym_DRQN_jobshop50
import numpy as np
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchrl.data import ReplayBuffer
from tensordict import TensorDict
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

class model(nn.Module):          
    def __init__(self):
        super(model,self).__init__()
        self.hidden_size = 20       
        self.rnn= nn.RNN(input_size=245, hidden_size=20,num_layers=1,batch_first=True) #Tanh
        #self.fc  = nn.Linear(80, 80)  
        #self.fc1 = nn.Linear(80, 80)
        self.fc = nn.Linear(80, 7) #fc2

    def init_hidden(self,batch_size):
        return (torch.zeros(1,batch_size, self.hidden_size)) #num_layers, batch, hidden_size

    def forward(self,x,hidden):
        x = x.reshape(x.shape[0], 4, 245)  #batch_size, seq_length, input_size
        x, h_0 = self.rnn(x,hidden)
        #x = nn.Tanh()(self.fc(x.contiguous().view(-1, 80)))   #instead use nn.Tanh()
        #x = nn.Tanh()(self.fc1(x))                         #makes more sense for negative rews
        #x = self.fc2(x)
        return self.fc(x.contiguous().view(-1, 80)) #x

def list2tensor(x):
    return torch.tensor(x).float()    
def tensor2list(a):
    return [int(x) for x in a.tolist()]

env = gym.make('process-job50')
policy = model()
target_net = model()
target_net.load_state_dict(policy.state_dict())
#target_net.eval()
#optimizer = optim.RMSprop(policy.parameters())
optimizer = optim.Adam(policy.parameters(), lr = 0.00005)  # 0.0001
criterion = F.cross_entropy #torch.nn.MSELoss() #F.smooth_l1_loss #F.kl_div

memory = 200#5000
storage = LazyMemmapStorage(memory,scratch_dir='/scratch1/')
buffer = ReplayBuffer(collate_fn = lambda x:x, storage=storage)
gamma = 0.7
EPS_START = 0.8 #00.6
EPS_END = -0.465
LAST_SOURCES = 200  #800  
EPS_DECAY = 125000   #20000 125000
WINDOW = 30

def addEpisode(prev,curr,act,reward):
    tens_dict = TensorDict({'prev':prev,'curr':curr,'action':act,'reward':reward},batch_size=[1])
    buffer.extend(tens_dict)

def trainNet(total_episodes):
    if total_episodes == 0:
        return
    sample = buffer.sample(1)
    inp = sample.get('prev')   
    target = sample.get('curr')
    actions = sample.get('action')  
    rew = sample.get('reward')

    targets = target.reshape(-1,4,245)    #-1, num_seq, input_len [[[],[],[]], [[],[],[]], ...]
    hs = targets.shape[0]                 #number of steps registered

    ccs = inp.reshape(-1,4,245)           # -1, num_seq, input_len [[[],[],[]], [[],[],[]], ...]
    hs1 = ccs.shape[0]                    #number of steps registered

    hidden = target_net.init_hidden(hs)
    qvals  = target_net(targets,hidden)    #we get the Q vals of s', from the target net

    actions = actions.type('torch.LongTensor')
    actions = actions.reshape(50,1)  

    hidden = policy.init_hidden(hs1)
    inps = policy(ccs,hidden).gather(1,actions)  #we get the Q vals of s from the policy net using the actions as indexes
    p1,p2 = qvals.detach().max(1)         # values and indices of max values
    targ = torch.Tensor(1,p1.shape[0])    #tensor with the number of qvalues (max) in p1

    #the individual qvalues of the target = reward + gamma * maxQval(s') 
    p1[-1] = 0
    targ[0] = rew + gamma*p1           #if we are in any other step
    #target[0][-1] = rew[-1]            # if we are in the last step of the episode 

    optimizer.zero_grad()
    inps = inps.reshape(1,50)  #we reshape qvalues(s) into (1,steps)
    loss = criterion(inps,targ)
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp(-1,1)     #clippiing values in the gradients
    optimizer.step()

def trainDRQN(episodes):
    max_steps = 50
    steps_done = 0
    global store
    eps_threshold = EPS_START
    #testRs = []
    #testDemand = []
    #proc_tim = []
    #action_list = []
    #demand_list = []
    #epsilon = []
    pbar = tqdm.trange(episodes) #new
    for epi in pbar:  
        prev = torch.zeros([4,245])  #seq_length, steps*(units+1)+#prods+1  (50*8+4+1)                
        p_time = torch.tensor([[[1,2,0,0,0,3,2],[1,0,2,0,0,3,2],[0,2,0,2,0,0,1],[2,0,0,0,2,0,2],[2,0,2,2,0,0,1],[1,2,0,0,0,0,1]],
                               [[1,2,0,0,0,2,2],[1,0,2,0,0,2,2],[0,2,0,2,0,0,1],[2,0,0,0,2,0,2],[2,0,2,2,0,0,1],[1,2,0,0,0,0,1]],
                               [[1,2,0,0,0,1,2],[1,0,2,0,0,1,2],[0,2,0,2,0,0,1],[2,0,0,0,2,0,2],[2,0,2,2,0,0,1],[1,2,0,0,0,0,1]]])

        #p_demand = [[8,5,6,5],[6,7,7,0],[6,8,6,7],[6,8,0,7]] 
        p_demand = torch.tensor([[12,9,10,9],[10,11,11,4],[10,12,10,11],[10,12,4,11]])
        a = random.choice([0,1,2])
        #a = 0
        times = p_time[a]
        #proc_tim.append(a)
        d1 = p_demand[random.choice([0,1,2,3])]
        d2 = p_demand[random.choice([0,1,2,3])]
        d3 = p_demand[random.choice([0,1,2,3])]
        #d1 = p_demand[0]
        #d2 = p_demand[1]
        #d3 = p_demand[2]
        d_i = torch.tensor([8,8,8,8])
        d_sum = sum((d1,d2,d3,d_i))
        #demand_list.append([d1,d2,d3])
        
        intimes = torch.zeros(WINDOW, dtype=torch.int64) #number of time steps
        done = False
        steps = 0
        rew = 0
        ol = 0
        w = 0
        s  = []
        s_ = []
        a  = []
        r  = []
        while done == False:
            if steps == 0:
                for i in range(len(prev)):
                    prev[i][-5:-1] = d_i                    
            elif steps == int(max_steps/5):
                prev[-1][-5:-1] += d1
            elif steps == int(max_steps*2/5):
                prev[-1][-5:-1] += d2
            elif steps == int(max_steps*3/5):
                prev[-1][-5:-1] += d3 

            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            steps+=1 
            hidden = policy.init_hidden(1)  #at every input in the RNN we have a hidden = zero tensor
            cat_seqs = torch.chunk(prev,4)
            cat_seqs = torch.cat(cat_seqs,1)
            output=policy(cat_seqs,hidden)      #instead of prev we introduce a window

            rand = random.uniform(0,1)
            if rand < eps_threshold:      #EPSILON
                action = random.choice([0,1,2,3,4,5,6])
            else:                          #GREEDY   
                h = prev[-1][-5:-1]
                if torch.all(h<0):
                    action = output.argmax().item() #select action only from non processed products 
                else:
                    j = [i for i,v in enumerate(h) if v > 0]
                    l = [i+1 for i in j]
                    k = [0]           #always 0 action
                    if 1 in l:
                        k.append(1)
                        k.append(2)
                    if 2 in l:
                        k.append(3)
                        k.append(4)
                    if 3 in l:
                        k.append(5)
                    if 4 in l:
                        k.append(6)
                    ix = output[0][k].argmax().item()
                    action = k[ix]                       
                #action = output.argmax().item()#select action only from non processed products 

            prev_o = torch.cat((prev[-1][8:-5], torch.zeros((8), dtype=int),prev[-1][-5:]))
            prev_os = tensor2list(prev_o)
            intimes = np.append(intimes[1:],(action))
            obs, reward, done, intimes, ovl1, ovl2 = env.step1(action, prev_os, intimes, times)
            rew = rew+reward
            ol = ol + ovl1
            w = w + ovl2
            obs = list2tensor([obs])
            obs = torch.cat((prev,obs),0)
            obs = obs[1:5]
            #to the replay buffer
            s.append(prev)
            s_.append(obs)
            a.append(action)
            r.append(reward) 

            #train policy
            trainNet(epi) #at every time step the net is updated

            #update window
            prev = obs
            steps_done+=1
            
        s  = torch.unsqueeze(torch.stack(s), dim=0)
        s_ = torch.unsqueeze(torch.stack(s_),dim=0)
        a  = torch.unsqueeze(torch.tensor(a),dim=0)
        r  = torch.unsqueeze(torch.tensor(r,dtype=torch.float),dim=0)
        addEpisode(s,s_,a,r)                
        #testRs += [rew]            
        dleft = sum(obs[-1][-5:-1])            
        d_sum = sum(d_sum)            
        dperc = torch.div(dleft,d_sum)*100   
        dperc = dperc.numpy()
        #testDemand.append(dperc) #[(sum(obs[-1][-5:-1])).item()]
        #epsilon += [eps_threshold]
        f = open('data_DRQN_jobshop50steps/dems_rews_ovs','a')
        writer = csv.writer(f,lineterminator='\n')
        writer.writerow([rew,dperc,ol,w])
        f.close()
        #Early stopping
        #if epi > 3000:
        #    if testRs[-5:] > [80,80,80,80,80]:
        #        break

        #update target network
        if epi % 10 == 0:
            target_net.load_state_dict(policy.state_dict())
        pbar.set_description("Episode counter") 

    pbar.close()
    #return testRs, testDemand, epsilon, action_list, proc_tim, demand_list 
    

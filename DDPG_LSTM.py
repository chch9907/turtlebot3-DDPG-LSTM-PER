#coding:utf-8
#!~/anaconda3/bin/python
import torch
import torch.nn as nn
import numpy as np
import rospy
import random
from torch.autograd import Variable # torch 中 Variable 模块
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 1001
batch_size = 64
TAU = 0.01
trace_len = 16
device = torch.device('cuda:0')
#PER parameters
prioritized_replay_alpha=0.6
prioritized_replay_beta0=0.4
prioritized_replay_beta_iters=None
prioritized_replay_eps=1e-6

class ANet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(ANet,self).__init__()
        #convolution layers
        #input size:(img_channels = 3, img_rows = 32, img_cols = 32 )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,  out_channels=16,  kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv1[0].weight.data.normal_(0, 0.5)  # initialization
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,   out_channels=32,  kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2[0].weight.data.normal_(0, 0.5)  # initialization
        self.Fc1 = nn.Sequential( 
            nn.Linear(in_features =130 ,out_features = 512),
            nn.LeakyReLU(inplace=True)
        )
        self.Fc1[0].weight.data.normal_(0, 0.5)  # initialization
        
        self.lstm = nn.LSTM(input_size =512 , hidden_size= 256,num_layers=1,batch_first=True)
        
        self.Fc2 = nn.Sequential(
            nn.Linear(in_features =256 ,out_features = 128),
            nn.LeakyReLU(inplace=True)
        )
        self.Fc2[0].weight.data.normal_(0, 0.5)  # initialization
        self.Fc3 = nn.Sequential(
          nn.Linear(in_features=128, out_features = s_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.Fc3[0].weight.data.normal_(0, 0.5)  # initialization
        #actor network
        # self.fc1 = nn.Linear(in_features=s_dim, out_features =32)  #s_dim = 8
        # self.fc1.weight.data.normal_(0, 0.5)  # initialization
        self.out = nn.Linear(in_features=s_dim, out_features =a_dim)  #a_dim = 2
        self.out.weight.data.normal_(0, 0.5)  # initialization

    def init_hidden(self,batch_size = 1):
        return (Variable(torch.zeros(1, batch_size, 256)).to(device), #hidden unit: (num_layers, batchsize, hidden_size)
                Variable(torch.zeros(1, batch_size, 256)).to(device)) #memory unit: (num_layers, batchsize, hidden_size)
        
    def forward(self,s_img, s_pos, hidden = None, batch_size = 1, seq_len = 1):
        # DDPG+LSTM input (batch_size*seq_len, channel, row, col)
        x = self.conv1(s_img)
        x = self.conv2(x)
        x = torch.reshape(x, (x.shape[0], -1)) #128 columns, 1 row
        # print x.size()
        x = torch.cat((x, s_pos), 1)  #concate in row dimension
        # print x.size()
        x= self.Fc1(x)
        #print x.shape
        # before go to RNN, reshape the input to (barch, seq, feature)
        x = x.reshape(batch_size, seq_len, -1)
        # print "input:",x.shape
        #x = x.permute(1, 0 , 2)
        hidden = self.init_hidden(batch_size) if hidden is None else hidden
        x, newhidden = self.lstm(x,hidden) # lstm input : (seq_len,batch_size, input_size)
        x = x.reshape(batch_size * seq_len, -1)
        x = self.Fc2(x)
        x = self.Fc3(x)

        #x = self.fc1(x)
        #x = torch.relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        action = x
        return action,newhidden

class CNet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(CNet,self).__init__()
        #convolution layers
        #input size:(img_channels = 3, img_rows = 32, img_cols = 32 )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,  out_channels=16,  kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv1[0].weight.data.normal_(0, 0.5)  # initialization
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,   out_channels=32,  kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv2[0].weight.data.normal_(0, 0.5)  # initialization
        self.Fc1 = nn.Sequential( 
            nn.Linear(in_features = 132,out_features = 512),
            nn.ReLU(inplace=True)
        )
        self.Fc1[0].weight.data.normal_(0, 0.5)  # initialization
        
        self.lstm = nn.LSTM(input_size =512, hidden_size= 256,num_layers=1,batch_first=True)
        
        self.Fc2 = nn.Sequential(
            nn.Linear(in_features =256 ,out_features = s_dim),
            nn.ReLU(inplace=True)
        )
        self.Fc2[0].weight.data.normal_(0, 0.5)  # initialization
        self.Fc3 = nn.Sequential(
            nn.Linear(in_features= s_dim, out_features = 32),
            nn.LeakyReLU(inplace=True)
        )
        self.Fc3[0].weight.data.normal_(0, 0.5)  # initialization 
        # critic network
        # self.fcs = nn.Linear(s_dim,32) #s_dim = 4
        # self.fcs.weight.data.normal_(0, 0.5)  # initialization
        # self.fca = nn.Linear(a_dim,32)  #a_dim = 2
        # self.fca.weight.data.normal_(0, 0.5)  # initialization
        self.out = nn.Linear(32,1)   
        self.out.weight.data.normal_(0, 0.5)  # initialization
        
    def init_hidden(self,batch_size = 1):
        return (Variable(torch.zeros(1, batch_size, 256)).to(device), #hidden unit: (num_layers, batchsize, hidden_size)
                Variable(torch.zeros(1, batch_size, 256)).to(device)) #memory unit: (num_layers, batchsize, hidden_size)
        
    def forward(self,s_img, s_pos,a,hidden = None, batch_size = 1, seq_len = 1):
        x = self.conv1(s_img)
        x = self.conv2(x)
        x = torch.reshape(x, (x.shape[0], -1)) #128 columns, 1 row
        x = torch.cat((a, x, s_pos), 1)  #concate in row dimension
        # print a.size()
        # print s_pos.size()
        # print x.size()
        x= self.Fc1(x) 
        # before go to RNN, reshape the input to (barch, seq, feature)
        x = x.reshape(batch_size, seq_len, -1)
        #x = x.permute(1, 0 , 2) #input:(seq_len, batch_size, exp_len)
        hidden = self.init_hidden(batch_size) if hidden is None else hidden
        x, newhidden = self.lstm(x,hidden)
        x = x.reshape(batch_size * seq_len, -1)
        x = self.Fc2(x)
        x = self.Fc3(x)
        # x = self.fcs(x)
        #net = torch.relu(x)
        actions_value = self.out(x)
        return actions_value,newhidden

class DDPG_LSTM(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super(DDPG_LSTM, self).__init__()  # by using 'DDPG(nn.Module)' and 'super(DDPG,self).__init__() instead of DDPG(object) can save the model parameters
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = np.zeros((MEMORY_CAPACITY, 1024 * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.Actor_eval = ANet(s_dim,a_dim)
        self.Actor_target = ANet(s_dim,a_dim)
        self.Critic_eval = CNet(s_dim,a_dim)
        self.Critic_target = CNet(s_dim,a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s_img,s_pos, hidden):
        s_img = torch.unsqueeze(torch.FloatTensor(s_img),0).to(device)
        s_pos = torch.unsqueeze(torch.FloatTensor(s_pos),0).to(device)
        a ,newhidden= self.Actor_eval.forward(s_img, s_pos, hidden)               #  ae（s）
        a = a.squeeze(0).detach()
        return a, newhidden

    def soft_update(self, net_target, net): # 'self' should be included in the parameters of function in the class
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)


    # def resetHidden(self,batchsize = 1):
    #     self.Actor_eval.hidden = self.Actor_eval.init_hidden(batch_size)
    #     self.Actor_target.hidden = self.Actor_target.init_hidden(batch_size)
    #     self.Critic_eval.hidden = self.Critic_eval.init_hidden(batch_size)
    #     self.Critic_target.hidden = self.Critic_target.init_hidden(batch_size)
    
    # def calculateTarget(self, br, done, q_):
    #     if done: 
    #         return br
    #     else:
    #         return br + GAMMA * q_
        
    def learn(self, Train_batch, trace_len, batch_size,prioritized, weights , eplison,batch_idxes):
        #Reset the recurrent layer's hidden state
        new_priorities = None
        h_ae = None
        h_ce = None
        h_at = None
        h_ct = None
        #Train_batch = Train_batch.permute(1,0,2) # trace_len * batch_size * experience_len
        bt =  torch.FloatTensor(Train_batch)  #(batch_size * trace_len , experience_len)
        #print bt.size()
        bs_img = torch.FloatTensor(bt[:, :1024* 3]).reshape(batch_size * trace_len,3, 32,32).to(device)  #img. channel, img.row, img.col
        #print bs_img.size()
        bs_pos = torch.FloatTensor(bt[:, 1024* 3:1024* 3 + 2]).reshape(batch_size * trace_len,-1).to(device)
        #print bs_pos.size()
        ba = torch.FloatTensor(bt[:, 1024* 3 + 2: 1024 * 3 + 2 + self.a_dim]).reshape(batch_size * trace_len,-1).to(device)
        #print ba.size()
        br = torch.FloatTensor(bt[:, -1024*3 - 6: -1024*3 - 5]).reshape(batch_size * trace_len,-1).to(device)
        #print br.size()
        bs_img_ = torch.FloatTensor(bt[:, -1024 * 3- 3: -3]).reshape(batch_size * trace_len, 3, 32,32).to(device)
        #print bs_img_.size()
        bs_pos_ = torch.FloatTensor(bt[:, -3:-1]).reshape(batch_size * trace_len,-1).to(device)
        bd = torch.FloatTensor(bt[:, -1:]).reshape(batch_size * trace_len,-1).to(device)
        
        weights = weights.reshape(-1, 1)
        weights = np.sqrt(weights)  #We do this since each weight will squared in MSE loss
        weights = torch.FloatTensor(weights).to(device)
        
        #In order to only propogate accurate gradients through the network, we will mask the first half of the losses for each trace as per Lample & Chatlot 2016
        maskA = torch.zeros([batch_size,trace_len //2])
        maskB = torch.ones([batch_size,trace_len //2]) 
        # concat two mask according to  column
        mask = torch.cat((maskA,maskB),1) 
        # reshape mask
        mask = torch.reshape(mask,[-1,1]) .to(device)
        #print mask
        
        #feed the image to the network
        a, h_ae = self.Actor_eval.forward(bs_img, bs_pos,  h_ae, batch_size, trace_len)
        q, h_ce = self.Critic_eval.forward(bs_img, bs_pos, a, h_ce, batch_size, trace_len)  # loss=-q=-ce（s,ae（s））   ae（s）=a   ae（s_）=a_
        loss_a = -torch.mean(q * mask) 
        self.atrain.zero_grad()
        loss_a.backward(retain_graph=True)
        self.atrain.step()         
               
        a_, h_at = self.Actor_target.forward(bs_img_,bs_pos_, h_at, batch_size, trace_len)
        q_, h_ct = self.Critic_target.forward(bs_img_, bs_pos_, a_, h_ct, batch_size, trace_len)
        q_target = br + GAMMA * q_ * (1 - bd).detach()
        q_v, h_ce = self.Critic_eval.forward(bs_img_, bs_pos_, ba, h_ce, batch_size, trace_len)
        q_target = q_target * mask
        q_v = q_v * mask
        # print q_target
        # print q_v
        #used for updating priorities of experiences
        TD_errors = (q_target - q_v).to(device)
        TD_errors = TD_errors.reshape(batch_size, trace_len)
        TD_errors = TD_errors.sum(axis = 1)
        #Weight TD errors 
        weighted_TD_errors = torch.mul(TD_errors, weights) #element-wise multiple
        #Create a zero tensor
        zero_tensor = torch.zeros(weighted_TD_errors.shape).to(device)
        #Compute critic loss, MSE of weighted TD_r
        critic_loss = self.loss_td(weighted_TD_errors,zero_tensor)
        
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba)    
        #TD_errors= self.loss_td(q_target ,q_v )  # critic: R+gama*Q（s',a）-Q（s，a）
        self.ctrain.zero_grad()
        critic_loss.backward()
        self.ctrain.step()
        
        #For prioritized exprience replay
        #Update priorities of experiences with TD errors
        if prioritized:
            td_errors = TD_errors.detach().data.cpu().numpy()
            #td_errors = td_errors.mean(axis=1)
            new_priorities = np.abs(td_errors) +eplison

        # soft target replacement written by myself
        self.soft_update(self.Critic_target, self.Critic_eval)
        self.soft_update(self.Actor_target, self.Actor_eval)
        return new_priorities

    
class experience_buffer():
    def __init__(self, buffer_size):
        # experience buffer initialization
        self.buffer = [] 
        # buffer size
        self.buffer_size = buffer_size 
        self._next_idx = 0
        
    # add element into buffer
    def add(self,experience):
        # #clear the old experience 
        # if len(self.buffer) + 1 >= self.buffer_size: 
        #     self.buffer[0:(1+len(self.buffer))-self.buffer_size] = [] 
        # # add the experience 
        # self.buffer.append(experience) 
        if len(experience) < trace_len:
            return
        if self._next_idx >= len(self.buffer):
            self.buffer.append(experience)
        else:
            self.buffer[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self.buffer_size
            
    # random sample a batch, 
    def sample(self, batch_size, trace_length):
        # random sample  batch_size  episodes  and each batch length is trace_len
        sampled_episodes = random.sample(self.buffer,batch_size) 
        sampledTraces = []
        for episode in sampled_episodes:
            # the starting point of a trace
            point = np.random.randint(0,len(episode)+1-trace_length) 
            # trace with len=trace_len
            sampledTraces.append(episode[point:point+trace_length]) 
        sampledTraces = np.array(sampledTraces)
        # reshape  traces  into  (batch_size * trace_len , -1)
        return np.reshape(sampledTraces,[batch_size* trace_length, -1]) 
    
    # random sample a batch with prioritized experience replay
    def sample_per(self, idxes, batch_size, trace_length):
        sampledTraces = []
        for i in idxes:
            episode = self.buffer[i]
            # the starting point of a trace
            point = np.random.randint(0,len(episode)+1-trace_length) 
             # trace with len=trace_len
            sampledTraces.append(episode[point:point+trace_length]) 
        sampledTraces = np.array(sampledTraces)
        # reshape  traces  into  (batch_size * trace_len , -1)
        return np.reshape(sampledTraces,[batch_size * trace_length, -1]) 


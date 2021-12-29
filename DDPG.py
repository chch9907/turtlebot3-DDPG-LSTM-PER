#coding:utf-8
#!~/anaconda3/bin/python
import torch
import torch.nn as nn
import numpy as np
import rospy

LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
TAU = 0.01
device = torch.device('cuda:0')

class ANet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(ANet,self).__init__()
        #convolution layers
        #input size:(img_channels = 4, img_rows = 32, img_cols = 32 )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4,  out_channels=16,  kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv1[0].weight.data.normal_(0, 0.5)  # initialization
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,   out_channels=32,  kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2[0].weight.data.normal_(0, 0.5)  # initialization
        self.Fc1 = nn.Sequential( 
            nn.Linear(in_features =130 ,out_features = 512),
            nn.LeakyReLU(inplace=True)
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Fc1[0].weight.data.normal_(0, 0.5)  # initialization
        self.Fc2 = nn.Sequential(
            nn.Linear(in_features =512 ,out_features = 128),
            nn.LeakyReLU(inplace=True)
        )
        self.Fc2[0].weight.data.normal_(0, 0.5)  # initialization
        self.Fc3 = nn.Linear(in_features=128, out_features = s_dim)
        self.Fc3.weight.data.normal_(0, 0.5)  # initialization
        #actor network
        self.fc1 = nn.Linear(in_features=s_dim, out_features =30)  #s_dim = 4
        self.fc1.weight.data.normal_(0, 0.5)  # initialization
        self.out = nn.Linear(in_features=30, out_features =a_dim)  #a_dim = 2
        self.out.weight.data.normal_(0, 0.5)  # initialization

    def forward(self,s_img, s_pos):
        x = self.conv1(s_img)
        x = self.conv2(x)
        x = torch.reshape(x, (x.shape[0], -1)) #2048 columns, 1 row
        x = torch.cat((x, s_pos), 1)  #concate in row dimension
        x= self.Fc1(x)
        x = self.Fc2(x)
        x = self.Fc3(x)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        actions_value = x
        return actions_value

class CNet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(CNet,self).__init__()
        #convolution layers
        #input size:(img_channels = 4, img_rows = 32, img_cols = 32 )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4,  out_channels=16,  kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv1[0].weight.data.normal_(0, 0.5)  # initialization
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,   out_channels=32,  kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2[0].weight.data.normal_(0, 0.5)  # initialization
        self.Fc1 = nn.Sequential( 
            nn.Linear(in_features = 132 ,out_features = 512),
            nn.LeakyReLU(inplace=True)
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Fc1[0].weight.data.normal_(0, 0.5)  # initialization
        self.Fc2 = nn.Sequential(
            nn.Linear(in_features =512 ,out_features = 128),
            nn.LeakyReLU(inplace=True)
        )
        self.Fc2[0].weight.data.normal_(0, 0.5)  # initialization
        self.Fc3 = nn.Linear(in_features=128, out_features = s_dim)
        self.Fc3.weight.data.normal_(0, 0.5)  # initialization 
        # critic network
        self.fcs = nn.Linear(s_dim,30) #s_dim = 4
        self.fcs.weight.data.normal_(0, 0.5)  # initialization
        self.fca = nn.Linear(a_dim,30)  #a_dim = 2
        self.fca.weight.data.normal_(0, 0.5)  # initialization
        self.out = nn.Linear(30,1)   
        self.out.weight.data.normal_(0, 0.5)  # initialization
        
    def forward(self,s_img,s_pos,a):
        x = self.conv1(s_img)
        x = self.conv2(x)
        x = torch.reshape(x, (x.shape[0], -1)) #2048 columns, 1 row
        x = torch.cat((a, x, s_pos), 1)  #concate in row dimension
        x= self.Fc1(x) 
        x = self.Fc2(x)
        x = self.Fc3(x)
        x = self.fcs(x)
        # y = self.fca(a)
        # net = torch.relu(x+y)
        net = torch.relu(x)
        actions_value = self.out(net)
        return actions_value


class DDPG(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super(DDPG, self).__init__()  # by using 'DDPG(nn.Module)' and 'super(DDPG,self).__init__() instead of DDPG(object) can save the model parameters
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = np.zeros((MEMORY_CAPACITY, 1024 * 4 * 2 + a_dim + 1 + 4), dtype=np.float32)
        self.pointer = 0
        self.Actor_eval = ANet(s_dim,a_dim)
        self.Actor_target = ANet(s_dim,a_dim)
        self.Critic_eval = CNet(s_dim,a_dim)
        self.Critic_target = CNet(s_dim,a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s_img,s_pos):
        s_img = torch.unsqueeze(torch.FloatTensor(s_img),0).to(device)
        s_pos = torch.unsqueeze(torch.FloatTensor(s_pos),0).to(device)
        a = self.Actor_eval(s_img, s_pos)[0].detach()                      #  ae（s）
        return a

    def soft_update(self, net_target, net): # 'self' should be included in the parameters of function in the class
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs_img = torch.FloatTensor(bt[:, :1024 * 4]).to(device)
        bs_pos = torch.FloatTensor(bt[:, 1024 * 4: 1024 * 4 + 2]).to(device)
        ba = torch.FloatTensor(bt[:, 1024 * 4 + 2: 1024 * 4 + self.a_dim + 2]).to(device)
        br = torch.FloatTensor(bt[:, -1024 * 4 - 3: -1024 * 4 - 2]).to(device)
        bs_img_ = torch.FloatTensor(bt[:, -1024 * 4 - 2: - 2]).to(device)
        bs_pos_ =  torch.FloatTensor(bt[:, -2:]).to(device)
        
        #reshape state to the image
        #bs = bs.resize(bs, (32,32))
        bs_img = bs_img.reshape(BATCH_SIZE,4, 32, 32)
        #bs_ = bs_.resize(bs_,(32,32))
        bs_img_ = bs_img_.reshape(BATCH_SIZE, 4, 32,32)
        
        #feed the image to the network
        a = self.Actor_eval(bs_img,bs_pos)
        q = self.Critic_eval(bs_img, bs_pos,a)  # loss=-q=-ce（s,ae（s））   ae（s）=a   ae（s_）=a_
        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_img_,bs_pos_)
        q_ = self.Critic_target(bs_img_, bs_pos_,a_)
        q_target = br + GAMMA * q_
        q_v = self.Critic_eval(bs_img, bs_pos,ba)
        td_error = self.loss_td(q_target,q_v)  # critic: R+gama*Q（s',a）-Q（s，a）
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

        # soft target replacement written by myself
        self.soft_update(self.Critic_target, self.Critic_eval)
        self.soft_update(self.Actor_target, self.Actor_eval)

    def store_transition(self, s_img, s_pos,a, r,  s_img_,s_pos_):
        s_img = s_img.reshape(-1) #1024 * 4
        s_img_ = s_img_.reshape(-1)#1024 * 4
        transition = np.hstack((s_img, s_pos,a, [r],  s_img_,s_pos_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

# if __name__ == '__main__' :
#     net = DDPG(4, 2, 1)
#     data_input = torch.randn(1, 32,32)
#     #net(data_input)

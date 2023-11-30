import numpy as np
import torch
from collide_detection import check_collision
from matplotlib import pyplot as plt

from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GraphConv, GATConv, GatedGraphConv, LEConv, APPNP ,global_mean_pool
from torch_geometric.data import Data, Batch

import random
ID = '2'

load = False
save = False
TRAIN = False

MaxStep = 0.5

class MyWrappr:
    def __init__(self):
        self.state = torch.tensor([[-1.0000, -1.0000], 
                                   [-0.2000, -1.0000], 
                                   [-0.6000,  0.3333], 
                                   [-1.0000, -0.3333], 
                                   [ 1.0000, -0.3333], 
                                   [ 0.6000,  1.0000]],dtype = torch.float,requires_grad=False)
        self.edge_index = torch.tensor([[0,0,0,1,1,1,2,2,2],
                                        [3,4,5,3,4,5,3,4,5]],dtype = torch.long,requires_grad=False)
        self.step_n = 0
    

    def reset(self):
        self.state = torch.tensor([[-1.0000, -1.0000], 
                                   [-0.2000, -1.0000], 
                                   [-0.6000,  0.3333], 
                                   [-1.0000, -0.3333], 
                                   [ 1.0000, -0.3333], 
                                   [ 0.6000,  1.0000]],dtype = torch.float, requires_grad=False)
        self.step_n = 0
        return self.state

    def step(self,action):
        self.step_n += 1
        self.state = self.state + action
        triangle_1 = self.state[0:3]
        triangle_2 = self.state[3:6]
        if check_collision(triangle_1, triangle_2):
            reward = 0.0
            over = False
        else:
            reward = 6.0-self.step_n
            over = True

        #限制最大步数    
        
        if self.step_n >= 5 and over == False:
            over = True
            reward = 0.0
        
        #reward -= action.square().sum()

        return self.state, reward, over

        
    def show(self):
        
        plt.figure()
        plt.fill(self.state[0:3,0].detach().numpy(), self.state[0:3,1].detach().numpy(), '#7FC080')
        plt.fill(self.state[3:6,0].detach().numpy(), self.state[3:6,1].detach().numpy(), '#B3B2FE')
        plt.show()

env = MyWrappr()

#env.show()

###############################################################################################

class GCN(torch.nn.Module):
    def __init__(self, n_features, n_outputs, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GraphConv(n_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.linear1 = Linear(hidden_channels, n_outputs)
        self.dropout = Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.linear1(x)
        x = MaxStep*torch.tanh(x)
        return x
    
class GCN_scalar(torch.nn.Module):
    def __init__(self, n_features,  hidden_channels):
        super(GCN_scalar, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GraphConv(n_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if load:
    model_action = torch.load('data/test'+ID+'/model_action.pt')
    model_value1 = torch.load('data/test'+ID+'/model_value1.pt')
    model_value2 = torch.load('data/test'+ID+'/model_value2.pt')
    model_action_delay = torch.load('data/test'+ID+'/model_action_delay.pt')
    model_value1_delay = torch.load('data/test'+ID+'/model_value1_delay.pt')
    model_value2_delay = torch.load('data/test'+ID+'/model_value2_delay.pt')

else:
    model_action = GCN(2, 2, 32)
    model_action_delay = GCN(2, 2, 32)
    model_action_delay.load_state_dict(model_action.state_dict())

    model_value1 = GCN_scalar(2, 32)
    model_value1_delay = GCN_scalar(2, 32)
    model_value1_delay.load_state_dict(model_value1.state_dict())

    model_value2 = GCN_scalar(2, 32)
    model_value2_delay = GCN_scalar(2, 32)
    model_value2_delay.load_state_dict(model_value2.state_dict())

optimizer_action = torch.optim.Adam(model_action.parameters(), lr=5e-4)
optimizer_value1 = torch.optim.Adam(model_value1.parameters(), lr=5e-3)
optimizer_value2 = torch.optim.Adam(model_value2.parameters(), lr=5e-3)

if load:
    optimizer_action_load = torch.load('data/test'+ID+'/optimizer_action.pt')
    optimizer_value1_load = torch.load('data/test'+ID+'/optimizer_value1.pt')
    optimizer_value2_load = torch.load('data/test'+ID+'/optimizer_value2.pt')

    optimizer_action.load_state_dict(optimizer_action_load.state_dict())
    optimizer_value1.load_state_dict(optimizer_value1_load.state_dict())
    optimizer_value2.load_state_dict(optimizer_value2_load.state_dict())



#model_action.to(device)
#model_action_delay.to(device)
#model_value1.to(device)
#model_value1_delay.to(device)
#model_value2.to(device)
#model_value2_delay.to(device)

##################################################################################

def play(show=False):
    data = []
    reward_sum = 0
    state = env.reset()
    over = False
    while not over:
        with torch.no_grad():
            action = model_action(state, env.edge_index)+(torch.rand(6,2)-0.5)*0.1
            next_state, reward, over = env.step(action)
        data.append((state, action, reward, next_state, over))
        reward_sum += reward
        state = next_state
        if show:
            env.show()
    return data, reward_sum
    
#play(True)[-1]

#数据池
class Pool:

    def __init__(self):
        self.pool = []

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, i):
        return self.pool[i]

    #更新动作池
    def update(self):
        #每次更新不少于N条新数据
        old_len = len(self.pool)
        while len(pool) - old_len < 200:
            self.pool.extend(play()[0])

        #只保留最新的N条数据
        self.pool = self.pool[-2_000:]

    #获取一批数据样本
    def sample(self):
        data = random.sample(self.pool, 1)
        
        state = data[0][0]
        #state = Batch.from_data_list([Data(x=i[0], edge_index=env.edge_index) for i in data])
        action = data[0][1]
        reward = data[0][2]
        next_state = data[0][3]
        over = data[0][4]
        return  state,action, reward, next_state, over


pool = Pool()
#pool.update()
#state, action, reward, next_state, over = pool.sample()
#print(state, reward.shape, pool[0])

#####################################################################################


def soft_update(_from, _to):
    for _from, _to in zip(_from.parameters(), _to.parameters()):
        value = _to.data * 0.7 + _from.data * 0.3
        _to.data.copy_(value)


def requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad_(value)


requires_grad(model_action_delay, False)
requires_grad(model_value1_delay, False)
requires_grad(model_value2_delay, False)

def train_action(state):
    requires_grad(model_action, True)
    requires_grad(model_value1, False)
    requires_grad(model_value2, False)

    #首先把动作计算出来
    action = model_action(state,env.edge_index)
    #使用value网络评估动作的价值,价值是越高越好
    input = state+action
    value1 = model_value1(input,env.edge_index)
    value2 = model_value2(input,env.edge_index)
    loss = -torch.min(value1, value2).mean()
    #print(value1.item(), value2.item())
    loss.backward()
    optimizer_action.step()
    optimizer_action.zero_grad()

    return loss.item()


#train_action(state)

def train_value(state, action, reward, next_state, over):
    requires_grad(model_action, False)
    requires_grad(model_value1, True)
    requires_grad(model_value2, True)

    #计算value
    with torch.no_grad():
        input = state+action
    value1 = model_value1(input,env.edge_index)
    value2 = model_value2(input,env.edge_index)
    
    #计算target
    next_action = model_action_delay(next_state,env.edge_index)
    
    with torch.no_grad():
        input = next_state+ next_action
        target1 = model_value1_delay(input,env.edge_index)
        target2 = model_value2_delay(input,env.edge_index)
    target = torch.min(target1, target2)
    target = target * 0.99 * (1 - over) + reward
    #计算td loss,更新参数
    loss1 = torch.nn.functional.mse_loss(value1, target)
    loss2 = torch.nn.functional.mse_loss(value2, target)

    loss1.backward()
    optimizer_value1.step()
    optimizer_value1.zero_grad()

    loss2.backward()
    optimizer_value2.step()
    optimizer_value2.zero_grad()

    return loss1.item(), loss2.item()


#train_value(state, action, reward, next_state, over)

#####################################################################################

def train():
    model_action.train()
    model_value1.train()
    model_value2.train()

    #共更新N轮数据
    for epoch in range(20*4):
        pool.update()

        #每次更新数据后,训练N次
        for i in range(1000):

            #采样N条数据
            state, action, reward, next_state, over = pool.sample()

            #训练模型
            train_action(state)
            
            #for _,param in enumerate(model_value1.named_parameters()):
            #    print(param[0])
            #    print(param[1])
            #    print('----------------')
            #print('----------------------------------**----------------------------------')
            train_value(state, action, reward, next_state, over)

        soft_update(model_action, model_action_delay)
        soft_update(model_value1, model_value1_delay)
        soft_update(model_value2, model_value2_delay)

        if epoch % 20 == 0:
            test_result = sum([play()[-1] for _ in range(20)]) / 20
            print(int(epoch/20), test_result)

if __name__ == '__main__':
    
    if TRAIN:
        train()
    if save:
        torch.save(model_action, 'data/test'+ID+'/model_action.pt')
        torch.save(model_value1, 'data/test'+ID+'/model_value1.pt')
        torch.save(model_value2, 'data/test'+ID+'/model_value2.pt')
        torch.save(model_action, 'data/test'+ID+'/model_action_delay.pt')
        torch.save(model_value1, 'data/test'+ID+'/model_value1_delay.pt')
        torch.save(model_value2, 'data/test'+ID+'/model_value2_delay.pt')
        torch.save(optimizer_action, 'data/test'+ID+'/optimizer_action.pt')
        torch.save(optimizer_value1, 'data/test'+ID+'/optimizer_value1.pt')
        torch.save(optimizer_value2, 'data/test'+ID+'/optimizer_value2.pt')
    print(play(True)[-1])
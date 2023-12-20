import socket
import time
import numpy as np
import trimesh
import torch
import random
import copy

from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GraphConv,global_mean_pool

objSrc = 'E:/siggraph2024/siggraph-2024/assets/obj/48.obj'
load = False
ID = '3.1'
MaxStep = 0.01
count_iter = 0
np.set_printoptions(threshold=np.inf,suppress=True)

class MyWrappr():
    def __init__(self):
        self.mesh = trimesh.load_mesh(objSrc)
        self.vertexs = torch.tensor(self.mesh.vertices,dtype=torch.float32,requires_grad=False)
        self.vertsNum = self.vertexs.shape[0]
        self.faces = torch.tensor(self.mesh.faces,dtype = torch.long,requires_grad=False)
        self.facesNum = self.faces.shape[0]

        self.vertexs_temp = torch.tensor(self.mesh.vertices,dtype=torch.float32,requires_grad=False)
        self.mask = torch.ones(self.vertsNum, dtype=torch.bool) #true 的地方需要被设置成0

        self.step_n = 0
        
        vertexs_str = np.array2string(self.vertexs.numpy(),precision=6).replace('[', '').replace(']', '')
        faces_str = np.array2string(self.faces.numpy()).replace('[', '').replace(']', '')
        
        print('vertexs string length : ',len(vertexs_str))

        
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(('127.0.0.1', 8080))
        self.server.listen(1)
        self.connection, self.address = self.server.accept()

        self.connection.send(bytes(vertexs_str, encoding="ascii"))
        time.sleep(0.1)

        self.connection.send(bytes(faces_str, encoding="ascii"))

        recv_str = self.connection.recv(1024)
        recv_str = recv_str.decode("ascii")
        
        num_list = []
        str_lst = recv_str.split(' ')
        for s in str_lst:
            if s.isdigit():
                num_list.append(int(s))

        self.collision_num_init = len(num_list)/2
        edge_num = int(9*len(num_list)/2)
        self.collision_edge_init = torch.empty((2,edge_num), dtype = torch.long,requires_grad=False)

        for i in range(int(len(num_list)/2)):
            id0 = num_list[2*i]
            id1 = num_list[2*i+1]
            for j in range(3):
                self.mask[self.faces[id0][j]] = False
                self.mask[self.faces[id1][j]] = False
                for k in range(3):
                    self.collision_edge_init[0][9*i+3*j+k] = self.faces[id0][k]
                    self.collision_edge_init[1][9*i+3*j+k] = self.faces[id1][j]

    def reset(self):
        self.step_n = 0
        self.vertexs_temp = copy.deepcopy(self.vertexs)
        return self.vertexs_temp,self.collision_edge_init,self.collision_num_init,self.mask

    def step(self,action):
        global count_iter
        count_iter+=1
        print('step : ',count_iter)
        self.vertexs_temp += action
        mask = torch.ones(self.vertsNum, dtype=torch.bool)
        #print(self.vertexs_temp)
        #time.sleep(0.01)
        vertexs_str = np.array2string(self.vertexs_temp.numpy(),precision=6).replace('[', '').replace(']', '')
        #print(vertexs_str)
        #vertexs_str += 'e'
        #print('vertexs string length : ',len(vertexs_str))
        response =self.connection.send(bytes(vertexs_str, encoding="ascii"))
        #print(response)

        recv_str = self.connection.recv(2048)
        recv_str = recv_str.decode("ascii")
        if recv_str == '!!!':
            collision_num == 0
        else:
            num_list = []
            str_lst = recv_str.split(' ')
            for s in str_lst:
                if s.isdigit():
                    num_list.append(int(s))

            collision_num = int(len(num_list)/2)
            edge_num = int(9*collision_num)
            collision_edge = torch.empty((2,edge_num), dtype = torch.long,requires_grad=False)

            for i in range(int(len(num_list)/2)):
                id0 = num_list[2*i]
                id1 = num_list[2*i+1]
                for j in range(3):
                    mask[self.faces[id0][j]] = False
                    mask[self.faces[id1][j]] = False
                    for k in range(3):
                        collision_edge[0][9*i+3*j+k] = self.faces[id0][k]
                        collision_edge[1][9*i+3*j+k] = self.faces[id1][j]
        
        self.step_n += 1
        if collision_num == 0:
            reward = 1
            over = True
        else:
            reward = -collision_num
            over = False
        if self.step_n >= 20 and over == False:
            over = True
            reward = -50

        return self.vertexs_temp ,collision_edge,mask, reward , over 
    
    def show(self):
        return 0
    

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
    model_action = GCN(3, 3, 32)
    model_action_delay = GCN(3, 3, 32)
    model_action_delay.load_state_dict(model_action.state_dict())

    model_value1 = GCN_scalar(3, 32)
    model_value1_delay = GCN_scalar(3, 32)
    model_value1_delay.load_state_dict(model_value1.state_dict())

    model_value2 = GCN_scalar(3, 32)
    model_value2_delay = GCN_scalar(3, 32)
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

########################################################################################################

env = MyWrappr()


def play(show=False,RANDOM=False):
    data = []
    state,edge,reward_sum,mask = env.reset()
    over = False
    while not over:
        with torch.no_grad():
            action = model_action(state, edge)#+(torch.rand(6,2)-0.5)*0.01
            if RANDOM:
                action += (torch.rand(env.vertsNum,3)-0.5)*0.001
            action[mask] = 0.0
            next_state,next_edge,next_mask ,reward_next, over = env.step(action)
        reward = reward_next-reward_sum  
        data.append((state, action, reward, next_state, over))
        state = next_state
        edge = next_edge
        mask = next_mask
        reward_sum = reward_next
        if show:
            env.show()
    return data, reward_sum

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
        while len(self.pool) - old_len < 200:
            self.pool.extend(play()[0])
            print(len(self.pool))
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
pool.update()

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

def train_action(state,edge):
    requires_grad(model_action, True)
    requires_grad(model_value1, False)
    requires_grad(model_value2, False)

    #首先把动作计算出来
    action = model_action(state,edge)
    #使用value网络评估动作的价值,价值是越高越好
    input = state+action
    value1 = model_value1(input,edge)
    value2 = model_value2(input,edge)
    loss = -torch.min(value1, value2).mean()
    #print(value1.item(), value2.item())
    loss.backward()
    optimizer_action.step()
    optimizer_action.zero_grad()

    return loss.item()


#train_action(state)

def train_value(state,edge ,action, reward, next_state, over):
    requires_grad(model_action, False)
    requires_grad(model_value1, True)
    requires_grad(model_value2, True)

    #计算value
    with torch.no_grad():
        input = state+action
    value1 = model_value1(input,edge)
    value2 = model_value2(input,edge)
    
    #计算target
    next_action = model_action_delay(next_state,edge)
    
    with torch.no_grad():
        input = next_state+ next_action
        target1 = model_value1_delay(input,edge)
        target2 = model_value2_delay(input,edge)
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

########################################################################################################

if __name__ == '__main__':
    print("start")

import numpy as np
import torch
from torch.nn import Linear, Dropout
from torch import tensor
from torch_geometric.nn import GCNConv, GraphConv, GATConv, GatedGraphConv, LEConv, APPNP
from torch_geometric.data.data import Data
from torch_geometric.data import DataLoader
from collide_detection import check_collision

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
        return x

model = GCN(2, 2, 32)
triangle1 = [(100, 100), (200, 100), (150, 200)]
triangle2 = [(100, 150), (350, 150), (300, 250)]
aabb_size= (200,250)
def play():
    state = torch.tensor([[100,100],[200,100],[150,200],[100,150],[350,150],[300,250]],dtype = torch.float)
    edge_index = torch.tensor([[0,1,2,3,4,3],
                               [1,2,0,4,5,5]],dtype = torch.long)
    data = Data(x=state, edge_index=edge_index)
    action = model(data.x, data.edge_index)
    next_state = state + action
    triangle_1 = state[0:3]
    triangle_2 = state[3:6]
    if check_collision(triangle_1, triangle_2):
        reward = -1
    else:
        reward = 1
    return state ,action ,next_state, reward 




def train():
    
    play()

if __name__ == '__main__':
    train()

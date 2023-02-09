import torch
import torch.nn as nn
import numpy as np
import logging
from coll_avo.methods.cadrl import PedestrianRL

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net

class ValueNetwork1(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        # 计算之前需要对坐标进行转换
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class ValueNetwork2(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp1 = mlp(input_dim, mlp1_dims)
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(mlp1_dims[-1], lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]

        state = torch.reshape(state, (-1, size[2]))
        mlp1_output = self.mlp1(state)
        mlp1_output = torch.reshape(mlp1_output, (size[0], size[1], -1))

        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(mlp1_output, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class LstmRL(PedestrianRL):
    def __init__(self):
        super().__init__()
        self.name = 'LSTM-RL'
        self.with_interaction_module = None
        self.interaction_module_dims = None

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('lstm_rl', 'mlp2_dims').split(', ')]
        global_state_dim = config.getint('lstm_rl', 'global_state_dim')
        self.with_om = config.getboolean('lstm_rl', 'with_om')
        with_interaction_module = config.getboolean('lstm_rl', 'with_interaction_module')
        if with_interaction_module:
            mlp1_dims = [int(x) for x in config.get('lstm_rl', 'mlp1_dims').split(', ')]
            self.model = ValueNetwork2(self.input_dim(), self.self_state_dim, mlp1_dims, mlp_dims, global_state_dim)
        else:
            self.model = ValueNetwork1(self.input_dim(), self.self_state_dim, mlp_dims, global_state_dim)
        self.multiagent_training = config.getboolean('lstm_rl', 'multiagent_training')
        logging.info('Method: {}LSTM-RL {} pairwise interaction module'.format(
            'OM-' if self.with_om else '', 'w/' if with_interaction_module else 'w/o'))

    def predict(self, state):
        #输入状态是机器人状态和行人可观测状态的联合，为了找到最佳动作,机器人对动作进行采样并执行,查看下一个状态的表现.

        def dist(pedestrian):
            # 排序行人(行人和机器人的距离)
            return np.linalg.norm(np.array(pedestrian.position) - np.array(state.self_state.position))

        state.pedestrian_states = sorted(state.pedestrian_states, key=dist, reverse=True)
        return super().predict(state)


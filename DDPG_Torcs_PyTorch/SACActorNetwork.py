import numpy as np
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
# 新增：SAC需要的分布模块
import torch.distributions as td

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(nn.Module):
    def __init__(self, state_size):
        super(ActorNetwork, self).__init__()
        # 【完全保留原有结构】前两层不变
        self.fc1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        
        # 【SAC修改】移除原有的确定性动作输出层
        # self.steering = nn.Linear(HIDDEN2_UNITS, 1)
        # self.acceleration = nn.Linear(HIDDEN2_UNITS, 1)
        # self.brake = nn.Linear(HIDDEN2_UNITS, 1)
        
        # 【SAC修改】新增：输出高斯分布的均值和对数标准差
        self.mean_layer = nn.Linear(HIDDEN2_UNITS, 3)
        self.log_std_layer = nn.Linear(HIDDEN2_UNITS, 3)
        
        # 【完全保留原有初始化】
        nn.init.normal_(self.mean_layer.weight, 0, 1e-4)
        nn.init.normal_(self.log_std_layer.weight, 0, 1e-4)
        self.mean_layer.bias.data.fill_(0.0)
        self.log_std_layer.bias.data.fill_(0.0)

    def forward(self, x):
        # 【完全保留原有逻辑】前两层激活不变
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 【SAC修改】输出分布参数
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = t.clamp(log_std, min=-20, max=2)  # 限制范围，避免数值爆炸
        
        return mean, log_std

    # 【SAC核心新增】重参数化采样 + 对数概率计算
    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal_dist = td.Normal(mean, std)
        
        # 重参数化采样：让采样过程可导
        x_t = normal_dist.rsample()
        y_t = t.tanh(x_t)  # 统一压缩到[-1,1]
        
        # 【适配TORCS动作范围】
        # steer: 保持[-1,1]
        # throttle/brake: 从[-1,1]映射到[0,1]
        steer = y_t[:, 0:1]
        throttle_brake = (y_t[:, 1:3] + 1.0) / 2.0
        action = t.cat([steer, throttle_brake], dim=1)
        
        # 计算对数概率，修正tanh的数值偏差
        log_prob = normal_dist.log_prob(x_t)
        log_prob -= t.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # 测试模式用的均值动作（无探索）
        mean_y_t = t.tanh(mean)
        mean_steer = mean_y_t[:, 0:1]
        mean_throttle_brake = (mean_y_t[:, 1:3] + 1.0) / 2.0
        mean_action = t.cat([mean_steer, mean_throttle_brake], dim=1)
        
        return action, log_prob, mean_action
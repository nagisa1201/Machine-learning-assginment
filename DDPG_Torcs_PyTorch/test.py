import torch
from torch.autograd import Variable
import numpy as np
import random
from gym_torcs import TorcsEnv
import argparse
import collections
import os
import io
import sys
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

os.environ['ALSOFT_DRIVERS'] = 'null'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
# ==========================================
# 1. 命令行参数与实验版本控制
# ==========================================
parser = argparse.ArgumentParser(description="DDPG TORCS Training and Evaluation")
parser.add_argument("--run-name", type=str, default=None, help="本次实验的名称 (默认: 自动生成时间戳)")
parser.add_argument("--train", type=int, default=1, choices=[0, 1], help="1: 开启训练, 0: 仅测试推理")
args = parser.parse_args()

# 🌟 关键修复：清空参数，防止底层的 gym_torcs 读取未知的参数而崩溃
sys.argv = [sys.argv[0]]

# 如果没有指定实验名，自动按当前时间生成 (例如: run_20231025_153022)
if args.run_name is None:
    run_name = "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
else:
    run_name = args.run_name

train_indicator = args.train

# 动态生成本次实验专属的文件夹路径
model_dir = os.path.join('model', run_name)
data_dir = os.path.join('data', run_name)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

print(f"\n🚀 [实验启动] 实验名称: {run_name}")
print(f"📁 模型将保存在: {model_dir}/")
print(f"📊 数据将保存在: {data_dir}/\n")

# ==========================================
# 2. 超参数设置
# ==========================================
state_size = 29
action_size = 3
LRA = 0.0001
LRC = 0.001
BUFFER_SIZE = 100000  
BATCH_SIZE = 32
GAMMA = 0.95
EXPLORE = 100000.
epsilon = 1 if train_indicator else 0 # 测试模式下直接关闭噪声探索
TAU = 0.001
VISION = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OU = OU()

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

# ==========================================
# 3. 指标记录与动态绘图类 (适配子文件夹)
# ==========================================
class MetricsLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.history = collections.defaultdict(list)
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 2, figsize=(15, 18))
        self.fig.tight_layout(pad=5.0)
        self.reset_episode_vars()

    def reset_episode_vars(self):
        self.ep_reward, self.ep_steps, self.ep_q_sum = 0, 0, 0
        self.ep_c_loss_sum, self.ep_a_loss_sum, self.ep_updates = 0, 0, 0
        self.prev_steer, self.jerk_sum = 0, 0
        self.overlap_count, self.speed_sum = 0, 0
        self.track_dev_sum, self.offtrack_count = 0, 0

    def step_record(self, action, ob, reward, q_val=None, c_loss=None, a_loss=None):
        steer, throttle, brake = action
        self.ep_reward += reward
        self.ep_steps += 1
        if q_val is not None:
            self.ep_q_sum += q_val
            self.ep_c_loss_sum += c_loss
            self.ep_a_loss_sum += a_loss
            self.ep_updates += 1

        self.jerk_sum += (steer - self.prev_steer) ** 2
        self.prev_steer = steer
        if throttle > 0.05 and brake > 0.05:
            self.overlap_count += 1
            
        self.speed_sum += getattr(ob, 'speedX', 0)
        self.track_dev_sum += getattr(ob, 'trackPos', 0) ** 2
        if abs(getattr(ob, 'trackPos', 0)) > 1.0:
            self.offtrack_count += 1

    def end_episode(self, episode, ob):
        avg_q = self.ep_q_sum / max(1, self.ep_updates)
        avg_c_loss = self.ep_c_loss_sum / max(1, self.ep_updates)
        avg_a_loss = self.ep_a_loss_sum / max(1, self.ep_updates)
        
        avg_jerk = self.jerk_sum / max(1, self.ep_steps)
        overlap_rate = self.overlap_count / max(1, self.ep_steps)
        avg_speed = (self.speed_sum / max(1, self.ep_steps)) * 300
        avg_track_dev = np.sqrt(self.track_dev_sum / max(1, self.ep_steps))
        damage = getattr(ob, 'damage', 0)

        self.history['Episode'].append(episode)
        self.history['Reward'].append(self.ep_reward)
        self.history['Steps'].append(self.ep_steps)
        self.history['Avg_Q'].append(avg_q)
        self.history['Critic_Loss'].append(avg_c_loss)
        self.history['Actor_Loss'].append(avg_a_loss)
        self.history['Steering_Jerk'].append(avg_jerk)
        self.history['Overlap_Rate'].append(overlap_rate)
        self.history['Avg_Speed(km/h)'].append(avg_speed)
        self.history['Track_Deviation'].append(avg_track_dev)
        self.history['Off_Track_Count'].append(self.offtrack_count)
        self.history['Damage'].append(damage)

        # 保存到专属的子文件夹中
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.save_dir, 'training_metrics.csv'), index=False)
        
        self.update_plot()
        self.reset_episode_vars()
        return self.ep_reward

    def update_plot(self):
        titles = ['Reward (Total)', 'Survival Steps', 'Avg Q-Value', 'Critic Loss', 
                  'Steering Jerk (Smoothness)', 'Overlap Rate (Throttle/Brake)', 
                  'Avg Speed (km/h)', 'Track Deviation (Center Offset)']
        keys = ['Reward', 'Steps', 'Avg_Q', 'Critic_Loss', 
                'Steering_Jerk', 'Overlap_Rate', 'Avg_Speed(km/h)', 'Track_Deviation']
        colors = ['blue', 'green', 'purple', 'red', 'orange', 'brown', 'cyan', 'magenta']

        for i, ax in enumerate(self.axs.flatten()):
            ax.clear()
            ax.plot(self.history['Episode'], self.history[keys[i]], color=colors[i])
            ax.set_title(titles[i], fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.pause(0.01)
        self.fig.savefig(os.path.join(self.save_dir, 'dynamic_metrics.png'))

# ==========================================
# 4. 网络初始化与模型加载
# ==========================================
actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)
critic = CriticNetwork(state_size, action_size).to(device)

print("正在检查历史模型权重...")
actor_path = os.path.join(model_dir, 'actormodel.pth')
critic_path = os.path.join(model_dir, 'criticmodel.pth')

try:
    actor.load_state_dict(torch.load(actor_path))
    critic.load_state_dict(torch.load(critic_path))
    actor.eval()
    critic.eval()
    print(f"✅ 成功从 {model_dir} 加载模型！")
except Exception as e:
    if not train_indicator:
        print(f"❌ 警告: 在测试模式下找不到模型文件 ({actor_path})，请检查 --run-name 是否正确！")
    else:
        print("🌱 未找到历史模型，随机初始化权重，从零开始训练。")

buff = ReplayBuffer(BUFFER_SIZE)

target_actor = ActorNetwork(state_size).to(device)
target_critic = CriticNetwork(state_size, action_size).to(device)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

criterion_critic = torch.nn.MSELoss(reduction='sum')
optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LRC)

env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor') 

# 实例化记录器，传入本次实验的 data 专属路径
logger = MetricsLogger(data_dir)

# ==========================================
# 5. 主循环
# ==========================================
for i in range(2000):
    if np.mod(i, 3) == 0:
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    
    for j in range(100000):
        if train_indicator:
            epsilon -= 1.0 / EXPLORE
            
        a_t = np.zeros([1, action_size])
        noise_t = np.zeros([1, action_size])
        
        a_t_original = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())

        if torch.cuda.is_available():
            a_t_original = a_t_original.data.cpu().numpy()
        else:
            a_t_original = a_t_original.data.numpy()

        noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
        noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
        noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

        if train_indicator and random.random() <= 0.1:
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0.2, 1.00, 0.10)
        
        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

        ob_new, r_t, done, info = env.step(a_t[0])
        s_t1 = np.hstack((ob_new.angle, ob_new.track, ob_new.trackPos, ob_new.speedX, ob_new.speedY, ob_new.speedZ, ob_new.wheelSpinVel/100.0, ob_new.rpm))

        buff.add(s_t, a_t[0], r_t, s_t1, done)
        batch = buff.getBatch(BATCH_SIZE)
        
        q_val, c_loss, a_loss = 0, 0, 0

        if len(batch) == BATCH_SIZE and train_indicator:
            states = torch.tensor(np.asarray([e[0] for e in batch]), device=device).float()
            actions = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
            rewards = torch.tensor(np.asarray([e[2] for e in batch]), device=device).float()
            new_states = torch.tensor(np.asarray([e[3] for e in batch]), device=device).float()
            dones = np.asarray([e[4] for e in batch])
            y_t = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
            
            target_q_values = target_critic(new_states, target_actor(new_states))
            q_val = target_q_values.mean().item()

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            q_values = critic(states, actions)
            loss = criterion_critic(y_t, q_values)  
            optimizer_critic.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_critic.step()
            c_loss = loss.item()

            a_for_grad = actor(states)
            a_for_grad.requires_grad_()
            q_values_for_grad = critic(states, a_for_grad)
            critic.zero_grad()
            q_sum = q_values_for_grad.sum()
            q_sum.backward(retain_graph=True)
            
            a_loss = -q_sum.item() / BATCH_SIZE

            grads = torch.autograd.grad(q_sum, a_for_grad) 
            act = actor(states)
            actor.zero_grad()
            act.backward(-grads[0])
            optimizer_actor.step()

            new_actor_state_dict = collections.OrderedDict()
            new_critic_state_dict = collections.OrderedDict()
            for var_name in target_actor.state_dict():
                new_actor_state_dict[var_name] = TAU * actor.state_dict()[var_name] + (1-TAU) * target_actor.state_dict()[var_name]
            target_actor.load_state_dict(new_actor_state_dict)

            for var_name in target_critic.state_dict():
                new_critic_state_dict[var_name] = TAU * critic.state_dict()[var_name] + (1-TAU) * target_critic.state_dict()[var_name]
            target_critic.load_state_dict(new_critic_state_dict)
        
        logger.step_record(a_t[0], ob_new, r_t, q_val, c_loss, a_loss)
        s_t = s_t1
        
        if done:
            break

    total_reward = logger.end_episode(i, ob_new)
    print(f"=====================================================")
    print(f"🏁 Episode {i} | Run: {run_name} | Reward: {total_reward:.2f} | Steps: {logger.history['Steps'][-1]}")
    print(f"   Avg Q: {logger.history['Avg_Q'][-1]:.2f} | Critic Loss: {logger.history['Critic_Loss'][-1]:.2f}")
    print(f"   Steer Jerk: {logger.history['Steering_Jerk'][-1]:.4f} | Avg Speed: {logger.history['Avg_Speed(km/h)'][-1]:.1f} km/h")
    print(f"=====================================================\n")

    if np.mod(i, 5) == 0 and train_indicator:
        # 保存带有特定回合数的文件，供以后选择最佳模型
        torch.save(actor.state_dict(), os.path.join(model_dir, f'actormodel_ep{i}.pth'))
        torch.save(critic.state_dict(), os.path.join(model_dir, f'criticmodel_ep{i}.pth'))
        # 永远维护一个最新的文件，便于中断后恢复
        torch.save(actor.state_dict(), os.path.join(model_dir, 'actormodel.pth'))
        torch.save(critic.state_dict(), os.path.join(model_dir, 'criticmodel.pth'))

env.end()
print("Finish.")
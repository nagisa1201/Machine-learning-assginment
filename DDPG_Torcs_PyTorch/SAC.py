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
import torch.nn.functional as F

from ReplayBuffer import ReplayBuffer
from SACActorNetwork import ActorNetwork
# 移除OU依赖
# from OU import OU

os.environ['ALSOFT_DRIVERS'] = 'null'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

# ==========================================
# 1. 命令行参数与实验版本控制 (100%保留你的逻辑)
# ==========================================
parser = argparse.ArgumentParser(description="SAC TORCS Training and Evaluation")
parser.add_argument("--run-name", type=str, default=None, help="本次实验的名称 (默认: 自动生成时间戳)")
parser.add_argument("--train", type=int, default=1, choices=[0, 1], help="1: 开启训练, 0: 仅测试推理")
args = parser.parse_args()

# 关键修复：清空参数，防止gym_torcs崩溃
sys.argv = [sys.argv[0]]

if args.run_name is None:
    run_name = "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
else:
    run_name = args.run_name

train_indicator = args.train

# 仅改文件夹前缀，区分SAC
model_dir = os.path.join('model_SAC', run_name)
data_dir = os.path.join('data_SAC', run_name)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

print(f"\n🚀 [SAC 实验启动] 实验名称: {run_name}")
print(f"📁 模型将保存在: {model_dir}/")
print(f"📊 数据将保存在: {data_dir}/\n")

# ==========================================
# 2. 超参数设置 (保留DDPG原有参数，新增SAC特有参数)
# ==========================================
state_size = 29
action_size = 3
LRA = 0.0001
LRC = 0.001
BUFFER_SIZE = 100000  
BATCH_SIZE = 32
GAMMA = 0.95
TAU = 0.001
VISION = False

# SAC新增超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_entropy = -action_size  # 目标熵，通常设为-action_size
lr_alpha = 3e-4
alpha = torch.tensor(0.2, device=device, requires_grad=True)  # 可学习的温度参数

# 移除OU
# OU = OU()

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

# ==========================================
# 3. 指标记录与动态绘图类
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
# 4. 网络初始化与模型加载 (SAC双Critic修改)
# ==========================================
# 这里的CriticNetwork需要你保留原有的文件，我们在主程序里实例化两个
# 请确保你的CriticNetwork.py和原DDPG版本一致（或按之前建议把输出层改为1）
from SACCriticNetwork import CriticNetwork

actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)

# 【SAC核心修改】实例化两个独立的Critic
critic1 = CriticNetwork(state_size, action_size).to(device)
critic2 = CriticNetwork(state_size, action_size).to(device)
critic1.apply(init_weights)
critic2.apply(init_weights)

# 对应的两个目标Critic
target_critic1 = CriticNetwork(state_size, action_size).to(device)
target_critic2 = CriticNetwork(state_size, action_size).to(device)
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())
target_critic1.eval()
target_critic2.eval()

# 优化器
criterion_critic = torch.nn.MSELoss(reduction='sum')
optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic1 = torch.optim.Adam(critic1.parameters(), lr=LRC)
optimizer_critic2 = torch.optim.Adam(critic2.parameters(), lr=LRC)
optimizer_alpha = torch.optim.Adam([alpha], lr=lr_alpha)

# 模型加载（适配双Critic）
print("正在检查历史模型权重...")
actor_path = os.path.join(model_dir, 'actormodel.pth')
critic1_path = os.path.join(model_dir, 'critic1model.pth')
critic2_path = os.path.join(model_dir, 'critic2model.pth')

try:
    actor.load_state_dict(torch.load(actor_path))
    critic1.load_state_dict(torch.load(critic1_path))
    critic2.load_state_dict(torch.load(critic2_path))
    actor.eval()
    critic1.eval()
    critic2.eval()
    print(f"✅ 成功从 {model_dir} 加载模型！")
except Exception as e:
    if not train_indicator:
        print(f"❌ 警告: 在测试模式下找不到模型文件，请检查 --run-name！")
    else:
        print("🌱 未找到历史模型，随机初始化权重，从零开始训练。")

# 初始化环境、经验池、日志
buff = ReplayBuffer(BUFFER_SIZE)
env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)
logger = MetricsLogger(data_dir)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor') 

# ==========================================
# 5. SAC 主训练循环 (保留你的环境交互逻辑，仅改算法核心)
# ==========================================
for i in range(2000):
    # 【100%保留你的TORCS重置逻辑】
    if np.mod(i, 3) == 0:
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    # 【100%保留你的状态预处理逻辑】不做归一化，和原DDPG一致
    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    
    for j in range(100000):
        # ==================================================
        # 【SAC修改1】动作选择：替换原DDPG+OU噪声
        # ==================================================
        a_t = np.zeros([1, action_size])
        state_tensor = torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float()
        
        if train_indicator:
            # 训练模式：采样带探索的动作
            action, log_prob, _ = actor.sample(state_tensor)
        else:
            # 测试模式：用均值动作，无探索
            _, _, mean_action = actor.sample(state_tensor)
            action = mean_action
        
        # 转换为numpy（和原逻辑一致）
        if torch.cuda.is_available():
            a_t_original = action.data.cpu().numpy()
        else:
            a_t_original = action.data.numpy()
        a_t[0] = a_t_original[0]

        # ==================================================
        # 【100%保留你的环境交互逻辑】
        # ==================================================
        ob_new, r_t, done, info = env.step(a_t[0])
        s_t1 = np.hstack((ob_new.angle, ob_new.track, ob_new.trackPos, ob_new.speedX, ob_new.speedY, ob_new.speedZ, ob_new.wheelSpinVel/100.0, ob_new.rpm))
        buff.add(s_t, a_t[0], r_t, s_t1, done)
        batch = buff.getBatch(BATCH_SIZE)
        
        # ==================================================
        # 【SAC修改2】网络更新逻辑
        # ==================================================
        q_val, c_loss, a_loss = 0, 0, 0
        if len(batch) == BATCH_SIZE and train_indicator:
            # 数据转换
            states = torch.tensor(np.asarray([e[0] for e in batch]), device=device).float()
            actions = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
            rewards = torch.tensor(np.asarray([e[2] for e in batch]), device=device).float()
            new_states = torch.tensor(np.asarray([e[3] for e in batch]), device=device).float()
            dones = torch.tensor(np.asarray([e[4] for e in batch]), device=device).float()

            # --------------------------
            # 1. 更新双Critic
            # --------------------------
            with torch.no_grad():
                next_actions, next_log_probs, _ = actor.sample(new_states)
                target_q1 = target_critic1(new_states, next_actions)
                target_q2 = target_critic2(new_states, next_actions)
                # SAC核心：取最小Q值抑制过估计，加熵项
                target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
                y_t = rewards + (1 - dones) * GAMMA * target_q.squeeze()

            # 更新Critic1
            q1 = critic1(states, actions).squeeze()
            loss1 = criterion_critic(q1, y_t)
            optimizer_critic1.zero_grad()
            loss1.backward(retain_graph=True)
            optimizer_critic1.step()

            # 更新Critic2
            q2 = critic2(states, actions).squeeze()
            loss2 = criterion_critic(q2, y_t)
            optimizer_critic2.zero_grad()
            loss2.backward(retain_graph=True)
            optimizer_critic2.step()

            # --------------------------
            # 2. 更新Actor
            # --------------------------
            current_actions, current_log_probs, _ = actor.sample(states)
            q1_actor = critic1(states, current_actions).squeeze()
            q2_actor = critic2(states, current_actions).squeeze()
            q_actor = torch.min(q1_actor, q2_actor)
            actor_loss = (alpha * current_log_probs.squeeze() - q_actor).mean()
            
            optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            optimizer_actor.step()

            # --------------------------
            # 3. 更新Alpha (温度参数)
            # --------------------------
            alpha_loss = -(alpha * (current_log_probs + target_entropy).detach()).mean()
            optimizer_alpha.zero_grad()
            alpha_loss.backward()
            optimizer_alpha.step()

            # 记录指标
            q_val = torch.min(q1, q2).mean().item()
            c_loss = (loss1.item() + loss2.item()) / 2
            a_loss = actor_loss.item()

            # --------------------------
            # 4. 软更新双目标Critic
            # --------------------------
            # 更新target_critic1
            for var_name in target_critic1.state_dict():
                target_critic1.state_dict()[var_name].data.copy_(
                    TAU * critic1.state_dict()[var_name].data + (1-TAU) * target_critic1.state_dict()[var_name].data
                )
            # 更新target_critic2
            for var_name in target_critic2.state_dict():
                target_critic2.state_dict()[var_name].data.copy_(
                    TAU * critic2.state_dict()[var_name].data + (1-TAU) * target_critic2.state_dict()[var_name].data
                )
        
        # ==================================================
        # 【100%保留你的指标记录逻辑】
        # ==================================================
        logger.step_record(a_t[0], ob_new, r_t, q_val, c_loss, a_loss)
        s_t = s_t1
        
        if done:
            break

    # ==================================================
    # 【100%保留你的日志打印和模型保存逻辑】
    # ==================================================
    total_reward = logger.end_episode(i, ob_new)
    print(f"=====================================================")
    print(f"🏁 Episode {i} | Run: {run_name} | Reward: {total_reward:.2f} | Steps: {logger.history['Steps'][-1]}")
    print(f"   Avg Q: {logger.history['Avg_Q'][-1]:.2f} | Critic Loss: {logger.history['Critic_Loss'][-1]:.2f}")
    print(f"   Steer Jerk: {logger.history['Steering_Jerk'][-1]:.4f} | Avg Speed: {logger.history['Avg_Speed(km/h)'][-1]:.1f} km/h")
    print(f"=====================================================\n")

    if np.mod(i, 5) == 0 and train_indicator:
        # 保存带回合数的模型
        torch.save(actor.state_dict(), os.path.join(model_dir, f'actormodel_ep{i}.pth'))
        torch.save(critic1.state_dict(), os.path.join(model_dir, f'critic1model_ep{i}.pth'))
        torch.save(critic2.state_dict(), os.path.join(model_dir, f'critic2model_ep{i}.pth'))
        # 保存最新模型
        torch.save(actor.state_dict(), os.path.join(model_dir, 'actormodel.pth'))
        torch.save(critic1.state_dict(), os.path.join(model_dir, 'critic1model.pth'))
        torch.save(critic2.state_dict(), os.path.join(model_dir, 'critic2model.pth'))

env.end()
print("Finish.")
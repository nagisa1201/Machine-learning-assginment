import torch
import torch.nn.functional as F
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
parser = argparse.ArgumentParser(description="TD3 TORCS Training and Evaluation")
parser.add_argument("--run-name", type=str, default=None, help="本次实验名称")
parser.add_argument("--train", type=int, default=1, choices=[0, 1], help="1: 训练, 0: 仅测试")
args = parser.parse_args()

sys.argv = [sys.argv[0]]

if args.run_name is None:
    run_name = "TD3_run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
else:
    run_name = args.run_name

train_indicator = args.train

model_dir = os.path.join('model_TD3', run_name)
data_dir = os.path.join('data_TD3', run_name)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

print(f"\n🚀 [TD3 实验启动] 实验名称: {run_name}")
print(f"📁 模型保存在: {model_dir}/")
print(f"📊 数据保存在: {data_dir}/\n")

# ==========================================
# 2. 超参数设置
# ==========================================
state_size = 29
action_size = 3
LRA = 0.0003          
LRC = 0.0003          
BUFFER_SIZE = 100000  
BATCH_SIZE = 256      
GAMMA = 0.99          
EXPLORE = 100000.
epsilon = 0.2 if train_indicator else 0
TAU = 0.001
VISION = False

# --- TD3专属参数 ---
POLICY_FREQ = 2         
POLICY_NOISE = 0.2      
NOISE_CLIP = 0.5        
WARMUP_STEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OU = OU()
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
# 4. TD3 网络初始化与模型加载
# ==========================================
actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)

critic1 = CriticNetwork(state_size, action_size).to(device)
critic2 = CriticNetwork(state_size, action_size).to(device)
critic1.apply(init_weights)
critic2.apply(init_weights)

print("正在检查历史模型权重...")
actor_path = os.path.join(model_dir, 'actormodel.pth')
critic1_path = os.path.join(model_dir, 'critic1model.pth')
critic2_path = os.path.join(model_dir, 'criti积累基础动力学数据.c2model.pth')

try:
    actor.load_state_dict(torch.load(actor_path))
    critic1.load_state_dict(torch.load(critic1_path))
    critic2.load_state_dict(torch.load(critic2_path))
    actor.eval()
    critic1.eval()
    critic2.eval()
    print(f"✅ 成功从 {model_dir} 加载 TD3 模型！")
except Exception as e:
    if not train_indicator:
        print(f"❌ 警告: 在测试模式下找不到模型文件，请检查 --run-name！")
    else:
        print("🌱 未找到历史模型，随机初始化权重，从零开始训练。")

buff = ReplayBuffer(BUFFER_SIZE)

target_actor = ActorNetwork(state_size).to(device)
target_critic1 = CriticNetwork(state_size, action_size).to(device)
target_critic2 = CriticNetwork(state_size, action_size).to(device)

target_actor.load_state_dict(actor.state_dict())
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=LRC)

env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

logger = MetricsLogger(data_dir)

# ==========================================
# 5. TD3 主循环
# ==========================================
total_it = 0 
for i in range(2000):
    if np.mod(i, 3) == 0:
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    low_speed_steps = 0  
    prev_steer = 0.0
    
    for j in range(100000):
        if train_indicator:
            epsilon -= 1.0 / EXPLORE
            
        a_t = np.zeros([1, action_size])
        
        # ==================================================
        # 受控热身，解决热身龟缩与过度撞墙
        # ==================================================
        if train_indicator and total_it < WARMUP_STEPS:
            a_t_original = np.zeros([1, action_size])
            a_t_original[0][0] = np.clip(np.random.normal(0.0, 0.2), -1.0, 1.0) 
            a_t_original[0][1] = np.random.uniform(0.2, 0.8)  
            if np.random.rand() < 0.1:
                a_t_original[0][2] = np.random.uniform(0.1, 0.5)
            else:
                a_t_original[0][2] = 0.0
        else:
            actor.eval()
            with torch.no_grad():
                a_t_original = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float()).cpu().numpy()
            actor.train()

        if train_indicator:
            noise_std = max(epsilon * 0.2, 0.05) 
            noise_steer = max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
            noise_accel = np.random.normal(0.0, noise_std) 
            noise_brake = np.random.normal(0.0, noise_std) 
            
            a_t[0][0] = np.clip(a_t_original[0][0] + noise_steer, -1.0, 1.0)
            a_t[0][1] = np.clip(a_t_original[0][1] + noise_accel, 0.0, 1.0)
            a_t[0][2] = np.clip(a_t_original[0][2] + noise_brake, 0.0, 1.0)
        else:
            a_t[0][0] = a_t_original[0][0]
            a_t[0][1] = a_t_original[0][1]
            a_t[0][2] = a_t_original[0][2]

        # 净加速度概念：处理物理踏板互斥，平滑梯度
        net_accel = a_t[0][1] - a_t[0][2]
        a_t[0][1] = np.clip(net_accel, 0.0, 1.0)  
        a_t[0][2] = np.clip(-net_accel, 0.0, 1.0) 

        # 执行动作
        ob_new, r_t, done, info = env.step(a_t[0])
        current_steer = a_t[0][0]
        prev_steer = current_steer 
        
        # ==================================================
        # 【破局点 3, 4, 5：终极奖励整形 (Reward Shaping)】
        # ==================================================
        if done:
            r_t = -300.0  
        else:
            v_kmh = ob_new.speedX * 300.0        
            angle_rad = ob_new.angle * np.pi     
            track_pos = ob_new.trackPos          
            v_effective = min(v_kmh, 90.0)
            
            progress_reward = v_effective * np.cos(angle_rad)
            track_penalty = (track_pos ** 2) * 30.0 
            angle_penalty = np.abs(np.sin(angle_rad)) * 30.0

            if v_kmh < 5.0:  
                slow_penalty = 5.0
            else:
                slow_penalty = 0.0
            
            # --- 破局点：基于距离的动态速度区间控制 ---
            front_dist = ob_new.track[9] if hasattr(ob_new, 'track') else 1.0
            corner_speed_penalty = 0.0
            
            # 假设 front_dist < 0.3 代表即将入弯或前方有墙（TORCS 中 1.0 通常代表 200 米，0.3 即 60 米）
            if front_dist < 0.3:
                target_min = 40.0
                target_max = 50.0
                
                if v_kmh > target_max:
                    # 速度过高：偏离越多，惩罚越重（二次方增长，严厉打击入弯不减速）
                    excess_speed = v_kmh - target_max
                    corner_speed_penalty = (excess_speed ** 2) * 0.05 
                elif v_kmh < target_min:
                    # 速度过低：给予线性惩罚，防止完全停下或过度龟缩
                    deficit_speed = target_min - v_kmh
                    corner_speed_penalty = deficit_speed * 0.5
                else:
                    # 完美落在 40-50 区间：给予奖励（在最终计算时表现为减去负值，即加分）
                    corner_speed_penalty = -20.0 
            
            # 如果在直道上（front_dist 很大），但速度极慢，依然保留你原来的慢速惩罚
            slow_penalty = 20.0 if (v_kmh < 5.0 and front_dist > 0.3) else 0.0
                
            understeer_penalty = (current_steer ** 2) * (v_kmh / 50.0) * 5.0
            
            # 更新最终的奖励计算
            r_t = progress_reward - track_penalty - angle_penalty - slow_penalty - corner_speed_penalty - understeer_penalty
            r_t = np.clip(r_t, -50.0, 100.0)
                
        r_t = r_t * 0.02
        # ==================================================

        s_t1 = np.hstack((ob_new.angle, ob_new.track, ob_new.trackPos, ob_new.speedX, ob_new.speedY, ob_new.speedZ, ob_new.wheelSpinVel/100.0, ob_new.rpm))

        buff.add(s_t, a_t[0], r_t, s_t1, done)
        batch = buff.getBatch(BATCH_SIZE)
        
        q_val, c_loss, a_loss = 0, 0, 0

        if len(batch) == BATCH_SIZE and train_indicator:
            total_it += 1
            
            states = torch.FloatTensor(np.asarray([e[0] for e in batch])).to(device)
            actions = torch.FloatTensor(np.asarray([e[1] for e in batch])).to(device)
            rewards = torch.FloatTensor(np.asarray([e[2] for e in batch])).unsqueeze(1).to(device)
            new_states = torch.FloatTensor(np.asarray([e[3] for e in batch])).to(device)
            dones = torch.FloatTensor(np.asarray([e[4] for e in batch])).unsqueeze(1).to(device)
            
            with torch.no_grad():
                next_action = target_actor(new_states)
                noise = torch.randn_like(next_action) * POLICY_NOISE
                noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
                next_action = next_action + noise
                
                # 【修复 1】: 计算 Target Q 时，也必须强制遵循物理踏板互斥逻辑
                next_action[:, 0] = next_action[:, 0].clamp(-1.0, 1.0)
                net_accel_tgt = next_action[:, 1] - next_action[:, 2]
                next_action[:, 1] = torch.clamp(net_accel_tgt, 0.0, 1.0)
                next_action[:, 2] = torch.clamp(-net_accel_tgt, 0.0, 1.0)

                target_Q1 = target_critic1(new_states, next_action)
                target_Q2 = target_critic2(new_states, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards + ((1.0 - dones) * GAMMA * target_Q)

            current_Q1 = critic1(states, actions)
            current_Q2 = critic2(states, actions)
            
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
            
            c_loss = critic_loss.item()
            q_val = current_Q1.mean().item()

            if total_it % POLICY_FREQ == 0:
                # 【修复 2】: 计算 Actor Loss 时，通过互斥过滤器让梯度反向传播
                raw_actions = actor(states)
                steer = raw_actions[:, 0:1]
                net_acc = raw_actions[:, 1:2] - raw_actions[:, 2:3]
                gas = torch.clamp(net_acc, 0.0, 1.0)
                brake = torch.clamp(-net_acc, 0.0, 1.0)
                filtered_actions = torch.cat([steer, gas, brake], dim=1)

                actor_loss = -critic1(states, filtered_actions).mean()
                
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()
                
                a_loss = actor_loss.item()

                for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        logger.step_record(a_t[0], ob_new, r_t, q_val, c_loss, a_loss)
        s_t = s_t1
        
        if done:
            break

    total_reward = logger.end_episode(i, ob_new)
    
    if total_it < WARMUP_STEPS and train_indicator:
        warmup_status = f"[受控热身: 积累基础动力学数据... {total_it}/{WARMUP_STEPS}]"
    else:
        warmup_status = ""
        
    print(f"=====================================================")
    print(f"🏁 Episode {i} | Run: {run_name} | Reward: {total_reward:.2f} | Steps: {logger.history['Steps'][-1]} {warmup_status}")
    print(f"   Avg Q: {logger.history['Avg_Q'][-1]:.2f} | Critic Loss: {logger.history['Critic_Loss'][-1]:.2f}")
    print(f"   Steer Jerk: {logger.history['Steering_Jerk'][-1]:.4f} | Avg Speed: {logger.history['Avg_Speed(km/h)'][-1]:.1f} km/h")
    print(f"=====================================================\n")

    if np.mod(i, 5) == 0 and train_indicator:
        torch.save(actor.state_dict(), os.path.join(model_dir, f'actormodel_ep{i}.pth'))
        torch.save(critic1.state_dict(), os.path.join(model_dir, f'critic1model_ep{i}.pth'))
        torch.save(critic2.state_dict(), os.path.join(model_dir, f'critic2model_ep{i}.pth'))
        
        torch.save(actor.state_dict(), os.path.join(model_dir, 'actormodel.pth'))
        torch.save(critic1.state_dict(), os.path.join(model_dir, 'critic1model.pth'))
        torch.save(critic2.state_dict(), os.path.join(model_dir, 'critic2model.pth'))

env.end()
print("Finish.")
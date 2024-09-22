import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
# 定义 Actor-Critic 网络，包含共享层
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 共享层，用于提取状态特征
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
        )
        # 策略层，输出每个动作的概率
        self.policy_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # 使用Softmax确保概率之和为1
        )
        # 价值层，输出状态的估计价值
        self.value_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出一个标量，表示状态价值
        )

    def forward(self, state):
        # 前向传播，输入状态，输出动作概率和状态价值
        shared = self.shared_layers(state)
        policy = self.policy_layers(shared)
        value = self.value_layers(shared)
        return policy, value

# PPO算法的超参数设置
clip_param = 0.2          # 策略剪辑参数
max_grad_norm = 0.5       # 梯度裁剪的最大范数
ppo_epochs = 10           # 每次更新的轮数
mini_batch_size = 64      # 小批量数据的大小
gamma = 0.99              # 折扣因子
lamda = 0.95              # GAE的lambda参数
entropy_coef = 0.01       # 熵损失的系数
value_loss_coef = 0.5     # 价值损失的系数
learning_rate = 3e-4      # 学习率

# 初始化Gym环境，这里使用月球着陆环境
env_name = 'LunarLander-v2'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]  # 状态空间的维度
action_dim = env.action_space.n             # 动作空间的大小

# 创建Actor-Critic模型和优化器
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 定义用于存储经验的缓冲区
class RolloutBuffer:
    def __init__(self):
        self.clear()  # 初始化时清空缓冲区

    def clear(self):
        # 用于存储状态、动作、概率、奖励等的列表
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

buffer = RolloutBuffer()  # 实例化缓冲区

# 用于绘制奖励曲线的列表
all_rewards = []          # 存储每个回合的总奖励
avg_rewards = []          # 存储最近N个回合的平均奖励
num_episodes = 1000       # 总的训练回合数
save_interval = 50        # 模型保存的间隔
video_interval = 200      # 录制视频的间隔

# 训练循环
for episode in range(num_episodes):
    state = env.reset()   # 重置环境，获取初始状态
    total_reward = 0      # 初始化当前回合的总奖励
    done = False          # 标志位，表示回合是否结束

    # 运行一个回合
    while not done:
        # 将状态转换为张量，并添加批次维度
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            # 获取动作概率和状态价值
            action_probs, value = model(state_tensor)
        # 根据动作概率创建分布
        dist = torch.distributions.Categorical(action_probs)
        # 从分布中采样动作
        action = dist.sample()
        # 获取动作的对数概率
        log_prob = dist.log_prob(action)
        
        # 在环境中执行动作，获取下一个状态、奖励等信息
        next_state, reward, done, _ = env.step(action.item())
        
        # 将经验存储到缓冲区
        buffer.states.append(state)
        buffer.actions.append(action)
        buffer.log_probs.append(log_prob)
        buffer.rewards.append(reward)
        buffer.is_terminals.append(done)
        buffer.values.append(value)

        # 更新状态和总奖励
        state = next_state
        total_reward += reward

    # 回合结束后，使用收集的经验更新策略
    # 计算折扣回报和优势

    # 计算折扣回报（Returns）
    rewards = []
    discounted_reward = 0
    # 反向遍历奖励序列
    for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
        if is_terminal:
            discounted_reward = 0  # 如果是终止状态，重置折扣回报
        discounted_reward = reward + (gamma * discounted_reward)
        rewards.insert(0, discounted_reward)  # 在列表开头插入折扣回报

    # 将列表转换为张量
    rewards = torch.tensor(rewards, dtype=torch.float32)
    values = torch.cat(buffer.values).detach().squeeze()

    # 计算优势（Advantages），A = R - V(s)
    advantages = rewards - values

    # 准备旧的数据用于PPO更新
    old_states = torch.tensor(buffer.states, dtype=torch.float32)
    old_actions = torch.cat(buffer.actions)
    old_log_probs = torch.cat(buffer.log_probs).detach()

    # 执行多次PPO更新
    for _ in range(ppo_epochs):
        # 创建小批量数据
        for index in range(0, len(buffer.states), mini_batch_size):
            # 采样小批量数据
            sampled_states = old_states[index:index+mini_batch_size]
            sampled_actions = old_actions[index:index+mini_batch_size]
            sampled_log_probs = old_log_probs[index:index+mini_batch_size]
            sampled_advantages = advantages[index:index+mini_batch_size]
            sampled_rewards = rewards[index:index+mini_batch_size]

            # 获取新的动作概率和状态价值
            action_probs, state_values = model(sampled_states)
            dist = torch.distributions.Categorical(action_probs)
            # 计算新的对数概率
            log_probs = dist.log_prob(sampled_actions)
            # 计算熵，增加探索
            entropy = dist.entropy().mean()

            # 计算新旧策略的概率比
            ratios = torch.exp(log_probs - sampled_log_probs)

            # 计算PPO的损失函数
            surr1 = ratios * sampled_advantages
            surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * sampled_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 计算价值函数的损失（均方误差）
            critic_loss = nn.MSELoss()(state_values.squeeze(), sampled_rewards)

            # 总损失，包括策略损失、价值损失和熵损失
            loss = actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    # 更新完毕后，清空缓冲区
    buffer.clear()

    # 记录总奖励，计算平均奖励
    all_rewards.append(total_reward)
    avg_reward = np.mean(all_rewards[-50:])  # 最近50个回合的平均奖励
    avg_rewards.append(avg_reward)
    print(f"Episode {episode+1}, Total Reward: {total_reward}, Average Reward: {avg_reward}")

    # 每隔一定回合保存模型
    if (episode + 1) % save_interval == 0:
        if not os.path.exists("model"):
            os.mkdir("model")
        torch.save(model.state_dict(), f"model/ppo_lunarlander_{episode+1}.pth")

    # 每隔一定回合录制智能体的游戏过程
    if (episode + 1) % video_interval == 0:
        # 导入RecordVideo包装器
        from gym.wrappers import RecordVideo
        # 创建新的环境用于录制，避免影响训练
        video_env = RecordVideo(env, video_folder="videos",
                                episode_trigger=lambda x: True, name_prefix=f"episode_{episode+1}")
        state = video_env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs, _ = model(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            state, reward, done, _ = video_env.step(action.item())
        video_env.close()

# 训练结束后，绘制奖励曲线并保存为图片
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('PPO on LunarLander-v2')
plt.savefig('reward_curve.png')
plt.show()

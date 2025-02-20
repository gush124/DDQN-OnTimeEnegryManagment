import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from distutils.version import LooseVersion
# Used for Atari
class Conv_Q(nn.Module):
	def __init__(self, frames, num_actions):
		super(Conv_Q, self).__init__()
		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.l1 = nn.Linear(3136, 512)
		self.l2 = nn.Linear(512, num_actions)


	def forward(self, state):
		q = F.relu(self.c1(state))
		q = F.relu(self.c2(q))
		q = F.relu(self.c3(q))
		q = F.relu(self.l1(q.reshape(-1, 3136)))
		return self.l2(q)


# Used for Box2D / Toy problems
class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_sizes=(128,256, 512, 256, 128)):
        super(FC_Q, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        input_size = state_dim
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.norms.append(nn.LayerNorm(hidden_size))  # 或 nn.LayerNorm(hidden_size)
            input_size = hidden_size
        self.output_layer = nn.Linear(input_size, num_actions)

    def forward(self, state):
        q = state
        for layer, norm in zip(self.layers, self.norms):
            q = F.relu(norm(layer(q)))  # Normalize after each layer
        return self.output_layer(q)


class DDQN(object):
	def __init__(
		self,
		num_actions,
		state_dim,
		device,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters=None,
		polyak_target_update=True,
		target_update_frequency=8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
		log_dir="runs/DDQN_experiment",
		decay_rate = 2000  # 新增指数衰减参数
	):
		self.log_dir = log_dir

		if optimizer_parameters is None:
			optimizer_parameters = {}

		# 初始化 TensorBoard 的 SummaryWriter

		self.writer = SummaryWriter(log_dir=self.log_dir)

		self.device = device

		# 初始化在线 Q 网络和目标 Q 网络
		self.Q = FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q).eval()
		for param in self.Q_target.parameters():
			param.requires_grad = False


		# 初始化优化器
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period
		self.decay_rate = decay_rate  # 新增指数衰减，主要是为了后期探索率更低，并且更快的达到低探索率

		# 如果是 Atari，state_dim 是 (C,H,W)，否则是向量长度
		self.state_shape = (-1, state_dim)
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Number of training iterations
		self.iterations = 0

	def select_action(self, state, eval=False):
		if eval:
			eps = self.eval_eps
		else:
			# 使用线性衰减公式，确保训练后期探索率低
			#eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)
			#使用指数衰减
			eps = self.end_eps + (self.initial_eps - self.end_eps) * np.exp(-1.0 * self.iterations / self.decay_rate)
		if np.random.rand() < eps:
			return np.random.randint(0, self.num_actions)
		else:
			state_t = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
			with torch.no_grad():
				q_values = self.Q(state_t)
				action = q_values.argmax(dim=1).item()
			return action

	def train(self, replay_buffer):
		# Sample replay buffer，不再返回 weights 和 indices
		state, action, next_state, reward, done = replay_buffer.sample()

		# Double DQN 的关键：
		# 1) 用在线网络挑选下一个动作( argmax Q_online )
		with torch.no_grad():
			next_actions = self.Q(next_state).argmax(dim=1, keepdim=True)

		# 2) 用目标网络估算目标 Q 值
		with torch.no_grad():
			target_Q_next = self.Q_target(next_state).gather(1, next_actions)
			# 如果 done=1，则后续没有未来奖励，所以乘 (1 - done)
			target_Q = reward + (1 - done) * self.discount * target_Q_next

		# 当前 Q 值
		current_Q = self.Q(state).gather(1, action)

		# 计算 TD-error，使用 L1 损失
		loss = F.l1_loss(current_Q, target_Q)

		# 优化
		self.Q_optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=10)  # 梯度裁剪
		self.Q_optimizer.step()

		# 更新目标网络
		self.iterations += 1
		self.maybe_update_target()

		# 打印日志
		if self.iterations % 1000 == 0:
			print(f"Iteration {self.iterations}: Loss = {loss.item()}")

		return loss

	# def train(self, replay_buffer):优先经验回放train
	#
	# 	#state, action, next_state, reward, done, indices, weights = replay_buffer.exp_sample()
	# 	# Sample replay buffer
	# 	state, action, next_state, reward, done, indices, weights = replay_buffer.sample()
	#
	# 	# Double DQN 的关键：
	# 	# 1) 用在线网络挑选下一个动作( argmax Q_online )
	# 	with torch.no_grad():
	# 		next_actions = self.Q(next_state).argmax(dim=1, keepdim=True)
	#
	# 	# 2) 用目标网络估算目标 Q 值
	# 	with torch.no_grad():
	# 		target_Q_next = self.Q_target(next_state).gather(1, next_actions)
	# 		# 如果 done=1，则后续没有未来奖励，所以乘 (1 - done)
	# 		target_Q = reward + (1 - done) * self.discount * target_Q_next
	#
	# 	# 当前 Q 值
	# 	current_Q = self.Q(state).gather(1, action)
	#
	# 	# 计算 TD-error
	# 	loss_elementwise = F.l1_loss(current_Q, target_Q, reduction='none')
	# 	loss = (loss_elementwise * weights).mean()
	#
	# 	td_error = (current_Q - target_Q).detach().abs().squeeze()
	# 	# 更新优先级
	# 	replay_buffer.update_priorities(indices, td_error.cpu().numpy())
	#
	# 	# 优化
	# 	self.Q_optimizer.zero_grad()
	#
	# 	loss.backward()
	# 	torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=10)  # 梯度裁剪
	# 	self.Q_optimizer.step()
	#
	# 	# 更新目标网络
	# 	self.iterations += 1
	# 	self.maybe_update_target()
	#
	# 	# 打印日志
	# 	if self.iterations % 1000 == 0:
	# 		print(f"Iteration {self.iterations}: Loss = {loss.item()}")
	#
	# 	return loss

	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			self.Q_target.load_state_dict(self.Q.state_dict())

	def save(self, filename):
		torch.save(self.Q.state_dict(), filename + "_Q")
		torch.save(self.Q_optimizer.state_dict(), filename + "_Q_optimizer")

	def load(self, filename):
		self.Q.load_state_dict(torch.load(filename + "_Q"))
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer.load_state_dict(torch.load(filename + "_Q_optimizer"))


	def close_writer(self):
		# 确保关闭 TensorBoard 的 Writer
		self.writer.close()
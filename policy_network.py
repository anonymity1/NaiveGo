import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys

# 检查是否为 macOS 且支持 MPS (M1/M2/M3)
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class Net(nn.Module):
    def __init__(self, width=19, height=19):
        super().__init__()

        self.width = width
        self.height = height

        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*width*height, width*height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*width*height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.width*self.height)
        # 注意: dim=1 是为了适配新版 PyTorch 的警告
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1) 
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.width*self.height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

class PolicyNetwork():
    def __init__(self, width=8, height=8, model_file=None, use_gpu=True): # use_gpu 参数保留但不再强制 CUDA
        self.board_width = width
        self.board_height = height
        
        # 1. 自动选择最佳设备 (Mac上为 mps 或 cpu)
        self.device = get_device()
        print(f"Running on device: {self.device}")

        self.policy_net = Net(width, height).to(self.device)

        self.l2_const = 1e-4
        self.optimizer = optim.Adam(self.policy_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            try:
                self.load_file(model_file)
            except FileNotFoundError:
                print(f"Warning: Model file {model_file} not found. Initializing with random weights.")

    def batch_policy_fn(self, board_state_batch):
        ''' 
        Input: a batch of board states
        Output: a batch of action probabilities, and state values
        '''
        # 2. 移除 Variable，使用 .to(device)
        state_batch = torch.tensor(board_state_batch, dtype=torch.float32).to(self.device)
        
        log_act_probs, value = self.policy_net(state_batch)
        
        # 3. 兼容 MPS/CPU 的 numpy 转换
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    def policy_fn(self, board):
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        
        state_tensor = torch.tensor(current_state, dtype=torch.float32).to(self.device)
        
        log_act_probs, value = self.policy_net(state_tensor)
        
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, board_state_batch, mcts_probs, winner_batch, lr=0.1):
        # 转换为 Tensor 并移动到设备
        state_batch = torch.tensor(board_state_batch, dtype=torch.float32).to(self.device)
        mcts_probs = torch.tensor(mcts_probs, dtype=torch.float32).to(self.device)
        winner_batch = torch.tensor(winner_batch, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()

        log_act_probs, value = self.policy_net(state_batch)
        
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, dim=1)) # 建议加 dim=1
        loss = value_loss + policy_loss

        loss.backward()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1))

        return loss.item(), entropy.item()

    def get_policy_param(self):
        return self.policy_net.state_dict()

    def save_model(self, model_file):
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)

    def load_file(self, model_file):
        # 4. 关键修改：map_location
        # 确保在 CPU/MPS 上也能加载别人用 CUDA 训练的模型
        net_params = torch.load(model_file, map_location='cpu') # 先加载到 CPU，再处理

        cur = self.board_height * self.board_width
        
        # 检查是否需要进行网络拼接（原始代码逻辑）
        if 'act_fc1.weight' in net_params and net_params['act_fc1.weight'].shape[1] != 4*cur:
            print("Try to apply the 8x8 model to a larger gomoku board!")
            a = torch.zeros(self.board_width*self.board_height, 4*self.board_width*self.board_height)
            b = torch.zeros(self.board_width*self.board_height)
            c = torch.zeros(64, 2*self.board_width*self.board_height)
            
            # 这里保持原逻辑，但需要注意维度的匹配
            a[(cur-64)//2:(cur+64)//2, 2*(cur-64):2*(cur+64)] = net_params['act_fc1.weight']
            b[(cur-64)//2:(cur+64)//2] = net_params['act_fc1.bias']
            c[:,cur-64:cur+64] = net_params['val_fc1.weight'] 
            
            net_params['act_fc1.weight'] = a
            net_params['act_fc1.bias'] = b
            net_params['val_fc1.weight'] = c
            
            # 设置不需要梯度，注意拼写 required -> requires
            # 但原始代码似乎是在加载后的 state_dict 上设置属性，这在 PyTorch 中是不起作用的
            # 我们只能在加载完模型后，对 self.policy_net 的参数设置 requires_grad
            # 为了保持原意，这里暂不改动逻辑，只加载修改后的参数

        self.policy_net.load_state_dict(net_params)
        # 将加载的参数移动到正确的设备 (mps/cpu)
        self.policy_net.to(self.device)

if __name__ == '__main__':
    # a = PolicyNetwork(10, 10,'./model/model100_10x10.model')
    # 确保文件路径存在，或者处理异常
    try:
        a = PolicyNetwork(10, 10, './best_model/best_model_8x8')
    except Exception as e:
        print(f"Error initializing: {e}")
        print("Creating a fresh model for testing...")
        a = PolicyNetwork(10, 10)
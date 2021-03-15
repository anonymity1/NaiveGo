import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
        self.act_fc1 = nn.Linear(4*width*height,
                                 width*height)
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
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.width*self.height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

class PolicyNetwork():
    def __init__(self, width=8, height=8, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = width
        self.board_height = height
        
        if self.use_gpu:
            self.policy_net = Net(width, height).cuda()
        else:
            self.policy_net = Net(width, height)

        self.l2_const = 1e-4
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), weight_decay=self.l2_const)

        # Initialize using the parameters in the file, 
        # or use pytorch's own method to initialize parameters.
        if model_file:
            self.load_file(model_file)

    def batch_policy_fn(self, board_state_batch):
        ''' 
        Input: a batch of board states
        Output: a batch of action probabilities, and state values
        '''
        if self.use_gpu:
            board_state_batch = Variable(torch.FloatTensor(board_state_batch).cuda())
            log_act_probs, value = self.policy_net(board_state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            board_state_batch = Variable(torch.FloatTensor(board_state_batch))
            log_act_probs, value = self.policy_net(board_state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value

    def policy_fn(self, board):
        '''Gomoku board state evaluation function, which is the core improvements in the AlphaZero version.

        Input: the board state, input size: 4*width*heigth.
        Output: probability of next move and the evaluation of the state of the board.
        '''
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, board_state_batch, mcts_probs, winner_batch, lr=0.1):
        '''This function is used to train the network.

        Input: board_state_batch, mcts_probs, winner_batch
        Output: the value of loss and entropy
        '''
        if self.use_gpu:
            board_state_batch = Variable(torch.FloatTensor(board_state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            board_state_batch = Variable(torch.FloatTensor(board_state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))
        
        self.optimizer.zero_grad()

        log_act_probs, value = self.policy_net(board_state_batch)
        # loss = (z-v)^2 - pi^T * log(p) + c * theta^2
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs))
        loss = value_loss + policy_loss

        loss.backward()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))

        return loss.item(), entropy.item()

    def get_policy_param(self):
        return self.policy_net.state_dict()

    def save_model(self, model_file):
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)

    def load_file(self, model_file):
        '''The smallest gomoku board size is 8 by 8.

        If the board size is bigger than 8 by 8 and you want to use some policy to the bigger gomoku board, 
        use the following code.
        '''
        net_params = torch.load(model_file) # 8 * 8 = 64

        cur = self.board_height * self.board_width
        if net_params['act_fc1.weight'].shape[1] != 4*cur:
            print("Try to apply the 8x8 model to a larger gomoku board!")
            a = torch.zeros(self.board_width*self.board_height, 4*self.board_width*self.board_height)
            b = torch.zeros(self.board_width*self.board_height)
            c = torch.zeros(64, 2*self.board_width*self.board_height)
            a[(cur-64)//2:(cur+64)//2, 2*(cur-64):2*(cur+64)] = net_params['act_fc1.weight']
            b[(cur-64)//2:(cur+64)//2] = net_params['act_fc1.bias']
            c[:,cur-64:cur+64] = net_params['val_fc1.weight'] 
            net_params['act_fc1.weight'] = a # cur, 4*cur
            net_params['act_fc1.bias'] = b # cur
            net_params['val_fc1.weight'] = c # 64, 2*cur

            net_params['conv1.weight'].required_grad = False
            net_params['conv1.bias'].required_grad = False
            net_params['conv2.weight'].required_grad = False
            net_params['conv2.bias'].required_grad = False
            net_params['conv3.weight'].required_grad = False
            net_params['conv3.bias'].required_grad = False
            net_params['act_conv1.weight'].required_grad = False
            net_params['act_conv1.bias'].required_grad = False

        self.policy_net.load_state_dict(net_params)

if __name__ == '__main__':
    # a = PolicyNetwork(10, 10,'./model/model100_10x10.model')
    a = PolicyNetwork(10, 10,'./best_model/best_model_8x8')

import numpy as np
import random

from collections import defaultdict, deque

from gomoku_board import Board
# from mcts import MCTS
from alpha import Alpha
from policy_network import PolicyNetwork

class Train():
    def __init__(self, init_model=None, row=8, column=8, 
                    batch_size=200, buffer_size=10000, epochs=5, game_batch_num=100, n_games=1, 
                    check_freq=10, lr_multiplier=1.0):
        self.row = row
        self.column = column
        self.init_model = init_model

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.epochs = epochs
        self.game_batch_num = game_batch_num
        self.n_games = n_games

        self.kl_targ = 0.02 # the target value of KL Divergence
        self.lr_multiplier = lr_multiplier # adjust the learning rate of the optimization algorithm
        self.check_freq = check_freq # check every few game rounds to see if the algorithm is improving

        self.policy_network = PolicyNetwork(model_file=init_model, width=column, height=row)
        self.best_policy_network = PolicyNetwork(model_file=init_model, width=column, height=row)

        # 确保 Alpha 类能接受一个现成的 network 实例
        self.AI = Alpha(policy_network=self.policy_network)
        self.best_AI = Alpha(policy_network=self.best_policy_network)

    # def get_equi_data(self, play_data):
    #     """数据增强：增加旋转和镜像数据"""
    #     extend_data = []
    #     for state, mcts_prob, winner in play_data:
    #         for i in [1, 2, 3, 4]:
    #             # 旋转 state (假设 state 维度是 [C, H, W])
    #             if self.row == self.column:
    #                 equi_state = np.array([np.rot90(s, i) for s in state])
    #                 # 旋转 mcts_prob (展平前需还原成二维)
    #                 equi_mcts_prob = np.rot90(mcts_prob.reshape(self.row, self.column), i)
    #                 extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
    #             # 水平翻转
    #             equi_state = np.array([np.fliplr(s) for s in equi_state])
    #             equi_mcts_prob = np.fliplr(equi_mcts_prob)
    #             extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
    #     return extend_data
    
    def get_equi_data(self, play_data):
        """
        数据增强：
        - 如果是正方形：增加 4 个旋转 x 2 个镜像 = 8 种变换
        - 如果是长方形：增加 旋转180度 + 水平翻转 + 垂直翻转 = 4 种变换
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            # 这里的 state 形状是 [C, row, col]
            # mcts_prob 形状是 [row * col]
            mcts_prob_2d = mcts_prob.reshape(self.row, self.column)
            
            # 1. 如果是正方形，可以进行全部 8 种变换
            if self.row == self.column:
                for i in [1, 2, 3, 4]:
                    # 旋转
                    equi_state = np.array([np.rot90(s, i) for s in state])
                    equi_mcts_prob = np.rot90(mcts_prob_2d, i)
                    extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
                    
                    # 翻转
                    equi_state_flip = np.array([np.fliplr(s) for s in equi_state])
                    equi_mcts_prob_flip = np.fliplr(equi_mcts_prob)
                    extend_data.append((equi_state_flip, equi_mcts_prob_flip.flatten(), winner))
            
            # 2. 如果是长方形，只能进行不改变维度的变换
            else:
                # 原图 (已经在外面 zip 过了，这里可以不加，或者显式加一次)
                # extend_data.append((state, mcts_prob, winner))
                
                # 水平翻转 (Left-Right Flip)
                st_lr = np.array([np.fliplr(s) for s in state])
                pr_lr = np.fliplr(mcts_prob_2d)
                extend_data.append((st_lr, pr_lr.flatten(), winner))
                
                # 垂直翻转 (Up-Down Flip)
                st_ud = np.array([np.flipud(s) for s in state])
                pr_ud = np.flipud(mcts_prob_2d)
                extend_data.append((st_ud, pr_ud.flatten(), winner))
                
                # 旋转 180 度 (相当于水平+垂直翻转)
                st_180 = np.array([np.rot90(s, 2) for s in state])
                pr_180 = np.rot90(mcts_prob_2d, 2)
                extend_data.append((st_180, pr_180.flatten(), winner))
                
        return extend_data
    
    def _collect_training_data(self):
        for i in range(self.n_games):
            self.AI.reset_root()

            self.board = Board(self.row, self.column)
            self.board.set_state()
            
            # 关键修改：不要在这里重新 reload 模型，使用外部持久化的 self.AI
            board_states, mcts_probs, current_players = [], [], []

            move_count = 0
            
            while True:
                # 建议在 Alpha 类中增加直接传入 board 对象的方法
                # 1. 动态调整温度 (Temperature)
                # 前 30 步（或棋盘较小时设为前 10-15 步）使用 temp=1.0，增加探索
                # 30 步之后使用 temp=1e-3，保证搜索质量
                temp = 1.0 if move_count < 15 else 1e-3

                move, move_probs = self.AI.self_play(self.row, self.column, self.board.board_state, temp=temp, is_self_play=True)
                
                board_states.append(self.board.current_state())
                mcts_probs.append(move_probs)
                current_players.append(self.board.get_cur_player())
                
                self.board.move(move)
                move_count += 1
                end, winner = self.board.who_win()
                
                if end:
                    winners = np.zeros(len(current_players))
                    if winner != 0:
                        winners[np.array(current_players) == winner] = 1.0
                        winners[np.array(current_players) != winner] = -1.0
                    
                    # 组合数据并进行增强
                    play_data = list(zip(board_states, mcts_probs, winners))
                    self.episode_len = len(play_data)
                    
                    # 存入增强后的数据
                    self.buffer.extend(self.get_equi_data(play_data))
                    break

    def _collect_training_data_old(self):
        '''Realize training data collection through self-play.'''
        for i in range(self.n_games):
            # generate self-play training data
            self.board = Board(self.row, self.column)
            self.board.set_state()
            AI = Alpha(model_file=self.init_model)
            board_states, mcts_probs, current_players = [], [], []
            while(True):
                move, move_probs = AI.self_play(self.row, self.column, self.board.board_state)
                board_states.append(self.board.current_state())
                mcts_probs.append(move_probs)
                self.board.move(move)
                current_players.append(self.board.get_cur_player())

                end, winner = self.board.who_win()
                if end:
                    winners = np.zeros(len(current_players))
                    if winner != 0:
                        winners[np.array(current_players) == winner] = 1.0
                        winners[np.array(current_players) != winner] = -1.0
                    print(winners)
                    play_data = zip(board_states, mcts_probs, winners)
                    break

            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # print(play_data)
            # add data to buffer
            self.buffer.extend(play_data)
            print(len(self.buffer))

    def _policy_update(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        board_state_batch = [x[0] for x in mini_batch]
        mcts_probs_batch = [x[1] for x in mini_batch]
        winner_batch = [x[2] for x in mini_batch]

        old_probs, old_v = self.policy_network.batch_policy_fn(board_state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_network.train_step(board_state_batch, mcts_probs_batch, winner_batch, lr=self.lr_multiplier)

            new_probs, new_v = self.policy_network.batch_policy_fn(board_state_batch)

            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 10:
                break
        
        if kl > self.kl_targ * 3 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.2
        else:
            self.lr_multiplier *= 1.2

        # explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))

        # explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        
        return loss, entropy

    def _eval_and_save_model_old(self, game_batch_num):
        '''Save model to the model directory.'''
        self.policy_network.save_model('./model/model{}_{}x{}.model'.format(game_batch_num+1, self.row, self.column))

    def _eval_and_save_model(self, game_batch_num, n_games=10):
        '''
        评估当前网络是否优于历史最优网络
        让 learner_ai 和 best_ai 对弈，平局各得 0.5 分
        '''
        print("--- Start Evaluating Current Model ---")
        win_cnt = defaultdict(int)
        for i in range(n_games):
            self.AI.reset_root()
            self.best_AI.reset_root()

            board = Board(self.row, self.column)
            board.set_state()

            learner_color = 1 if i % 2 == 0 else -1

            # 交替先后手
            if learner_color == 1:
                agents = {1: self.AI, -1: self.best_AI}
            else:
                agents = {1: self.best_AI, -1: self.AI}
            
            while True:
                # 4. 获取当前该走棋的颜色 (1 或 -1)
                curr_color = board.get_cur_player()
                move = agents[curr_color].get_action(board)

                # 5. 执行落子
                board.move(move)
                
                # 6. 同步更新两个 AI 的搜索树（如果 get_action 内部没更新的话）
                self.AI.update_with_move(move)
                self.best_AI.update_with_move(move)

                # 7. 胜负检查
                end, winner = board.who_win()
                if end:
                    if winner == learner_color:
                        win_cnt['learner'] += 1
                    elif winner == 0:
                        win_cnt['draw'] += 1
                    else:
                        win_cnt['best'] += 1
                    break
        
        win_rate = (win_cnt['learner'] + 0.5 * win_cnt['draw']) / n_games
        print(f"Win Rate: {win_rate:.2f} (Learner {win_cnt['learner']} vs Best {win_cnt['best']}, Draw {win_cnt['draw']})")
        if win_rate > 0.51:
            print("New best model found! Updating best model.")
            self.policy_network.save_model('./model/model{}_{}x{}.model'.format(game_batch_num+1, self.row, self.column))
        return win_rate

    def run(self):
        '''play the number of self.game_batch_num games.

        If the amount of data in the buffer is greater than batch_size, 
        perform one policy update including several epochs of training steps.
        '''

        # 增加目录检查，防止保存失败
        import os
        if not os.path.exists('./model'): os.makedirs('./model')

        try:
            for i in range(self.game_batch_num):
                self._collect_training_data()
                print("Game batch i: {}, episode_len:{}".format(i+1, self.episode_len))
                if len(self.buffer) > self.batch_size:
                    loss, entropy = self._policy_update()
                    print("Game batch i: {}. loss: {}. entropy: {}".format(i+1, loss, entropy))
                if (i+1) % self.check_freq == 0:
                    print("current self-play game batch i: {}".format(i+1))
                    self._eval_and_save_model(i)

        except KeyboardInterrupt:
            print('\n\rquit!')

if __name__ == '__main__':
    # train = Train(init_model='./best_model/best_model_8x8', row=8, column=8, batch_size=300, buffer_size=10000)
    # train = Train(init_model='./model/model100_10x10.model',row=13, column=13, batch_size=100, buffer_size=20000, check_freq=100)
    train = Train(row=13, column=13, batch_size=100, buffer_size=20000, game_batch_num=250, check_freq=10)
    train.run()

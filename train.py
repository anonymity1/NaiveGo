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
                    check_freq=10, lr_multiplier=1.0, use_gpu=True):
        self.row = row
        self.column = column
        self.init_model = init_model
        self.policy_network = PolicyNetwork(model_file=init_model, width=column, height=row)

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.epochs = epochs
        self.game_batch_num = game_batch_num
        self.n_games = n_games

        self.kl_targ = 0.02 # the target value of KL Divergence
        self.lr_multiplier = lr_multiplier # adjust the learning rate of the optimization algorithm
        self.check_freq = check_freq # check every few game rounds to see if the algorithm is improving

        self.use_gpu = use_gpu

    def _collect_training_data(self):
        '''Realize training data collection through self-play.'''
        for i in range(self.n_games):
            # generate self-play training data
            self.board = Board(self.row, self.column)
            self.board.set_state()
            AI = Alpha(model_file=self.init_model, use_gpu=self.use_gpu)
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
            if kl > self.kl_targ * 4:
                break
        
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        else:
            self.lr_multiplier *= 1.5

        # explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))

        # explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        
        return loss, entropy

    def _eval_and_save_model(self, game_batch_num):
        '''Save model to the model directory.'''
        self.policy_network.save_model('./model/model{}_{}x{}.model'.format(game_batch_num+1, self.row, self.column))

    def run(self):
        '''play the number of self.game_batch_num games.

        If the amount of data in the buffer is greater than batch_size, 
        perform one policy update including several epochs of training steps.
        '''
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
    train = Train(init_model='./best_model/best_model_10x10', row=10, column=10, batch_size=300, buffer_size=10000)
    train.run()

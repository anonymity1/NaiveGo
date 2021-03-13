import numpy as np
import random

from collections import defaultdict, deque

from gomoku_board import Board
# from mcts import MCTS
from alpha import Alpha
from policy_network import PolicyNetwork

class Train():
    def __init__(self, init_model=None, row=8, column=8, batch_size=20, buffer_size=10000):
        self.row = row
        self.column = column

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.epochs = 5
        self.n_games = 1

        self.kl_targ = 0.02
        self.game_batch_num = 10

        self.policy_network = PolicyNetwork(model_file=init_model, width=column, height=row)

        self.lr_multiplier = 1.0

        self.check_freq = 2

    def _collect_data(self, n_games=1):
        '''Realize training data collection through self-play.'''
        for i in range(n_games):
            # generate self-play training data
            self.board = Board(self.row, self.column)
            self.board.set_state()
            AI = Alpha(model_file='best_policy_pytorch.model')
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

    def run(self):
        '''play the number of self.game_batch_num games.

        If the amount of data in the buffer is greater than batch_size, 
        perform one policy update including several epochs of training steps.'''
        try:
            for i in range(self.game_batch_num):
                self._collect_data(self.n_games)
                print("batch i: {}, episode_len:{}".format(i+1, self.episode_len))
                if len(self.buffer) > self.batch_size:
                    loss, entropy = self._policy_update()
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    self.policy_network.save_model('./model{}.model'.format(i+1))

        except KeyboardInterrupt:
            print('\n\rquit!')

if __name__ == '__main__':
    train = Train()
    train.run()
    # a = [[1, 2, 3], [2, 3, 1]]
    # b = [[2, 7, 5], [3, 5, 2]]
    # c = [1, 2]
    # print(np.var(c))
    # a = np.array(a)
    # b = np.array(b)
    # res1 = a * (np.log(a + 1) - np.log(b + 1))
    # print(res1)
    # print(a)
    # res2 = np.sum(a, axis=0)
    # print(res2)
    # kl1 = np.mean(np.sum(a * (np.log(a + 1) - np.log(b + 1)), axis=(0,1)))
    # kl2 = np.mean(np.sum(a * (np.log(a + 1) - np.log(b + 1)), axis=0))
    # kl3 = np.mean(np.sum(a * (np.log(a + 1) - np.log(b + 1))))
    # print(kl1, kl2, kl3)
    # train._collect_data()
    # board = Board(8,8)
    # board.set_state()
    # board.move(2)
    # board.move(3)
    # board.move(4)
    # print(board)

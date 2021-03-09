import numpy as np
import random

from collections import defaultdict, deque

from gomoku_board import Board
# from mcts import MCTS
from alpha import Alpha
from policy_network import PolicyNetwork

class Train():
    def __init__(self, init_model=None, row=8, column=8, batch_size=512, buffer_size=10000):
        self.row = row
        self.column = column

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.epochs = 5
        self.n_games = 1

        self.kl_targ = 0.02
        self.game_batch_num = 100

        self.policy_network = PolicyNetwork(model_file=init_model, width=column, height=row)

    def _collect_data(self, n_games=1):
        for i in range(n_games):
            # generate self-play training data
            self.board = Board(self.row, self.column)
            self.board.set_state()
            AI = Alpha()
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
            # self.data_buffer.extend(training_data)

    def _policy_update(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        board_state_batch = [x[0] for x in mini_batch]
        mcts_probs_batch = [x[1] for x in mini_batch]
        winner_batch = [x[2] for x in mini_batch]
        old_probs, old_v = self.policy_network.batch_policy_fn(board_state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_network.train_step(board_state_batch, mcts_probs_batch, winner_batch)

            new_probs, new_v = self.policy_network.batch_policy_fn(board_state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        return loss, entroy

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self._collect_data(self.n_games)
                print("batch i: {}, episode_len:{}".format(i+1, self.episode_len))
                if len(self.buffer) > self.batch_size:
                    loss, entropy = self._policy_update()
        except KeyboardInterrupt:
            print('\n\rquit!')

if __name__ == '__main__':
    train = Train()
    train.run()
    train._collect_data()
    board = Board(8,8)
    board.set_state()
    board.move(2)
    board.move(3)
    board.move(4)
    print(board)

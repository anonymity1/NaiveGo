import numpy as np
import copy

from operator import itemgetter

C_PUCT=5.0

class Board():
    '''Gomukou Board state contains the following information:
    Board position where the current player can move.
    The impact of the move operation on the board.
    '''
    def __init__(self, row=19, column=19):
        '''The initialized board can only call the integer and coordinate conversion function'''
        self.row = row
        self.column = column

        self.last_move = -1
        self.is_black = True
        self.availables = list(range(self.row * self.column))

    def set_state(self, board_state:list):
        self.board_state = board_state
        cur = 0
        for x in range(self.row):
            for y in range(self.column):
                if board_state[x][y] == 0:
                    continue
                self.availables.remove(self.coordinate_to_integer(x,y))       
                if board_state[x][y] == 1:
                    cur = cur + 1
                else: # board_state[x][y] == -1
                    cur = cur - 1

        self.is_black = True if cur == 0 else False

    def move(self, action: int):
        self.last_move = action
        self.availables.remove(action)
        self.is_black = not self.is_black

        x, y = self.interger_to_coordinate(action)
        self.board_state[x][y] = self._ternary_op(1, -1, self.is_black)

    def coordinate_to_integer(self, x, y):
        return self.column * x + y

    def interger_to_coordinate(self, move):
        return move // self.column, move % self.column

    def who_win(self):
        '''TODO: last_move'''
        if self.last_move == -1:
            return False, 0
        x, y = self.interger_to_coordinate(self.last_move)
        four_dir = []
        four_dir.append([self.board_state[i][y] for i in range(self.row)])
        four_dir.append([self.board_state[x][j] for j in range(self.column)])
        def tilt_dir(x, y, dx, dy):
            cur = []
            while 0 <= x < self.row and 0 <= y < self.column:
                x, y = x + dx, y + dy
            x, y = x - dx, y - dy
            while 0 <= x < self.row and 0 <= y < self.column:
                cur.append(self.board_state[x][y])
                x, y = x - dx, y - dy
            return cur
        four_dir.append(tilt_dir(x, y, 1, 1))
        four_dir.append(tilt_dir(x, y, 1, -1))

        tag = self._ternary_op(1, -1, self.is_black)
        for l in four_dir:
            cnt = 0
            for p in l:
                if p == tag:
                    cnt += 1
                    if cnt == 5:
                        return True, self._ternary_op(1, -1, self.is_black)
                else:
                    cnt = 0
        if self.availables == []:
            return True, 0
        return False, 0

    def get_cur_player(self):
        return self.is_black

    def _ternary_op(self, black, white, tag:bool):
        return black if tag == True else white

class Node():
    '''A node in the Monte Carlo Search Tree contains the following information
    Relationship among different nodes, parents and children.
    '''
    def __init__(self, parent, prob:float):
        self.parent = parent
        self.children = {} # key-value pairs between actions and nodes

        # information about selected actions

        self.visited_num = 0
        self._Q = 0
        self._u = 0
        self._P = prob

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

    def get_value(self):
        self._u = (C_PUCT * self._P * np.sqrt(self.parent.visited_num) / (1 + self.visited_num))
        # print(self._Q + self._u)
        return self._Q + self._u

    def add_child(self, action:int, prob:float):
        self.children[action] = Node(self, prob)

class MCTS():
    '''Monte Carlo Tree and corresponding search algorithm'''
    def __init__(self):
        self.root = Node(None, 1.0)
        self.playout_num = 10
    
    def _policy(self, board):
        '''Output the probability of different positions according to the board information.
        And this mcts algorithm use a naive policy.

        Input: Board state
        Output: An iterator of (action, probability) and a score for the current board state.
        '''
        action_probs = np.ones(len(board.availables)) / len(board.availables)
        print(action_probs)
        return zip(board.availables, action_probs), 0

    def _select_best(self, node: Node):
        return max(node.children.items(), key=lambda a: a[1].get_value())

    def _expand(self, node: Node, action_probs: iter):
        ''''''
        for action, prob in action_probs:
            if action not in node.children:
                node.add_child(action, prob)

    def _update_recursive(self, node, leaf_value):
        pass

    def _evaluate_rollout(self, board):
        '''return -1 or 0 or 1'''
        player = board.get_cur_player()
        for i in range(limit):
            end, winner = board.game_end()
            if end:
                break
            # action_probs = 
        pass

    def _playout(self, board: Board):
        node = self.root
        while(True):
            if node.is_leaf():
                break
            action, node = self._select_best(node)
            board.move(action)

        action_probs, _ = self._policy(board)
        end, winner = board.who_win()
        print("tuoshi")        
        if not end:
            self._expand(node, action_probs)
        # leaf_value = self._evaluate_rollout(board)
        # self._update_recursive(node, -leaf_value)

    def play(self, row:int, column:int, board_state:list):
        '''AI exposed interface'''
        board = Board(row, column)
        board.set_state(board_state)

        for n in range(self.playout_num):
            board_copy = copy.deepcopy(board)
            self._playout(board_copy)
        move = max(self.root.children.items(), key=lambda a: a[1].visited_num)[0]

        x, y = board.interger_to_coordinate(move)
        return x, y

if __name__ == '__main__':
    board_state = [[0 for x in range(8)] for x in range(8)]
    mcts = MCTS()
    x, y = mcts.play(8, 8, board_state)
    print(x, y)


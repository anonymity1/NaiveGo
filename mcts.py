import numpy as np
import copy

from operator import itemgetter
from gomoku_board import Board

C_PUCT=5.0

class Node():
    '''A node in the Monte Carlo Search Tree contains the following information
    Relationship among different nodes, parents and children.
    '''
    def __init__(self, parent, prob:float):
        '''Initalize the Gomoku Board
        '''
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

    def update_value(self, leaf_value):
        self.visited_num += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self.visited_num

    def add_child(self, action:int, prob:float):
        self.children[action] = Node(self, prob)

class MCTS():
    '''Monte Carlo Tree and corresponding search algorithm'''
    def __init__(self):
        self.root = Node(None, 1.0)
        self.playout_num = 100
    
    def _policy(self, board):
        '''Output the probability of different positions according to the board information.
        And this mcts algorithm use a naive policy. In the AlphaZero version, use the Policy 
        Network to replace this naive function.

        Input: Board state
        Output: An iterator of (action, probability) and a score for the current board state.
        '''
        action_probs = np.ones(len(board.availables)) / len(board.availables)
        return zip(board.availables, action_probs), 0

    def _select_best(self, node: Node):
        '''Select best node among the child nodes according to the node's Q value.'''
        return max(node.children.items(), key=lambda a: a[1].get_value())

    def _expand(self, node: Node, action_probs: iter):
        '''If the game is not over in the current situation, expand this best child node'''
        for action, prob in action_probs:
            if action not in node.children:
                node.add_child(action, prob)

    def _random_rollout(self, board:Board):
        '''In this mcts algorithm, randomly select the child node. Don't expand, just search.
        
        This function is only called by the _evaluate_rollout function
        '''
        action_probs = np.random.rand(len(board.availables))
        return zip(board.availables, action_probs)

    def _evaluate_rollout(self, board, limit=300):
        '''Evaluate the current node by randomly expanding the current node.Return -1 or 0 or 1.

        In the AlphaZero version, we replace this simulation process with the policy_function.
        '''
        player = board.get_cur_player()
        for i in range(limit):
            end, winner = board.who_win()
            if end:
                break
            action_probs = self._random_rollout(board)
            max_action = max(action_probs, key=itemgetter(1))[0]
            board.move(max_action)
        else:
            print("Game is not over")
        return winner   

    def _update_recursive(self, node, leaf_value):
        '''Update the ancestry tree based on the result of the _evaluate_rollout function'''
        if not node.is_root():
            self._update_recursive(node.parent, -leaf_value)
        node.update_value(leaf_value)     

    def _playout(self, board: Board):
        node = self.root
        while(True):
            if node.is_leaf():
                break
            action, node = self._select_best(node)
            board.move(action)

        action_probs, _ = self._policy(board) # In AlphaZero version, _ means leaf_value

        end, winner = board.who_win()
        # Discuss in two situations: not the end game and the end game
        if not end:
            self._expand(node, action_probs)
            leaf_value = self._evaluate_rollout(board)
        else:
            if winner == 0:
                leaf_value = 0
            else:
                leaf_value = 1.0 if winner == board.get_cur_player() else -1.0
                # TODO: cur_player

        self._update_recursive(node, -leaf_value)
        # why -leaf_value? 
        # While calling the _select_best function, we use the child node Q value 
        # to decide which child node should be selected. But the current node and 
        # the child node belong different players.

        # self._show_tree(self.root, 1)

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

    def _show_tree(self, node:Node, cnt:int):
        '''For debugging only'''
        if node.is_leaf():
            return
        if node.is_root():
            print("root:", cnt)
        cnt += 1
        for index, child in node.children.items():
            print(index, ":", cnt, end='  ')
        print()
        for index, child in node.children.items():
            self._show_tree(child, cnt)

if __name__ == '__main__':
    board_state = [[0 for x in range(8)] for x in range(8)]
    mcts = MCTS()
    x, y = mcts.play(8, 8, board_state)
    print(x, y)


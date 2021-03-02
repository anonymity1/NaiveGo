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

        self.players = [1, -1]

    def set_state(self, board_state:list, start_player=0):
        '''Set Gomoku pieces information. Also need to set current player: 
        1 means black and -1 means white'''
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

        # Set current player:
        self.cur_player = self.players[start_player] 

        self.is_black = True if cur == 0 else False

    def move(self, action: int):
        '''Move pieces'''
        self.last_move = action
        self.availables.remove(action)
        self.is_black = not self.is_black

        x, y = self.interger_to_coordinate(action)
        self.board_state[x][y] = self._ternary_op(1, -1, self.is_black)

        self.cur_player = - self.cur_player # switch the current player

    def coordinate_to_integer(self, x, y):
        return self.column * x + y

    def interger_to_coordinate(self, move):
        return move // self.column, move % self.column

    def who_win(self):
        '''Determine which side wins the current state'''
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

        tag = self._ternary_op(1, -1, not self.is_black)
        for l in four_dir:
            cnt = 0
            for p in l:
                if p == tag:
                    cnt += 1
                    if cnt == 5:
                        return True, self._ternary_op(1, -1, not self.is_black)
                else:
                    cnt = 0
        if self.availables == []:
            return True, 0
        return False, 0

    def get_cur_player(self):
        '''get the current player in the current state'''
        return self.cur_player

    def _ternary_op(self, black, white, tag:bool):
        return black if tag == True else white

    def __str__(self):
        print(self.availables)
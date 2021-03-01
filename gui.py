from tkinter import * 
from tkinter.messagebox import *

import threading

from mcts import MCTS
# TODO: mcts and alphago algorithm
# from alpha import Alpha

class Gomoku():
    def __init__(self, row=19, column=19):
        self.row, self.column = row, column
        self.mesh = 25
        self.half_mesh = self.mesh / 2
        self.piece_r = 0.9 * self.half_mesh
        
        self.f_header_bg = "#e5e1ed"
        self.c_board_color = "#e6b953"
        self.b_font_info = ("楷体", 12, "bold")

        self.last_x, self.last_y = -1, -1
        self.is_black = True
        self.is_start = False
        self.human = False
        self.board = [[0 for x in range(self.column)] for x in range(self.row)]

    def run(self):
        self.root = Tk()
        self.root.title("Gomoku: Human vs AI")
        self.root.resizable(height = None, width = None)

        self.f_header = Frame(self.root, bg=self.f_header_bg)
        self.f_header.pack(fill=BOTH, ipadx=100)

        self.b_start_black = Button(self.f_header, text="执黑", command=self._start_black, font=self.b_font_info)
        self.b_start_white = Button(self.f_header, text="执白", command=self._start_white, font=self.b_font_info)
        self.l_info = Label(self.f_header, text="准备开始", bg=self.f_header_bg, font=("楷体", 14, "bold"), fg="grey")
        self.b_restart = Button(self.f_header, text="重开", command=self._restart, font=self.b_font_info, state=DISABLED)
        self.b_defeat = Button(self.f_header, text="认输", command=self._defeat, font=self.b_font_info, state=DISABLED)

        self.b_start_black.pack(side=LEFT, padx=20)
        self.b_start_white.pack(side=LEFT)
        self.l_info.pack(side=LEFT, expand=YES, fill=BOTH)
        self.b_defeat.pack(side=RIGHT, padx=20)
        self.b_restart.pack(side=RIGHT)

        self.c_gomoku = Canvas(self.root, bg=self.c_board_color, highlightthickness=0, width=(self.column+1)*self.mesh, height=(self.row+1)*self.mesh)
        self._state_shift(1)
        self._draw_board()
        self.c_gomoku.bind("<Button-1>", self._c_click)
        self.c_gomoku.pack()

        #1 TODO: board
        #2 TODO: click event
        self.root.mainloop()

    def _start_black(self):
        self.is_black = True
        self._state_shift(0)

    def _start_white(self):
        self.is_black = True
        self._state_shift(0)
        # self._AI_player()
        self._ai_thread()

    def _restart(self):
        self._state_shift(1)

    def _defeat(self):
        self._state_shift(2)
        self.c_gomoku.create_text(int(self.c_gomoku['width'])/2, int(self.c_gomoku['height'])/2, text="真菜", font=("楷体", 50, "bold"), fill='blue')

    def _state_shift(self, l:int):
        '''The header buttons can be seen as a state machine'''
        self.is_start = False
        self.human = False
        if l == 0:
            state_list = [DISABLED, DISABLED, NORMAL, NORMAL]
            self.is_start = True
            self.human = True
        elif l == 1:
            state_list = [NORMAL, NORMAL, DISABLED, DISABLED]
            self._draw_board()
            self.l_info.config(text='准备开始')
            self.board = [[0 for x in range(self.column)] for x in range(self.row)]
        elif l == 2:
            state_list = [DISABLED, DISABLED, NORMAL, DISABLED]
            self.is_start = False
        else: 
            # TODO: Exception handling
            return
        self.b_start_black.config(state=state_list[0])
        self.b_start_white.config(state=state_list[1])
        self.b_restart.config(state=state_list[2])
        self.b_defeat.config(state=state_list[3])

    def _draw_board(self):
        '''draw the whole board'''
        for x in range(self.column):
            for y in range(self.row):
                self._draw_mesh(x, y)
        # _draw_mesh

    def _draw_mesh(self, x, y):
        '''draw one grid'''
        self.c_gomoku.create_rectangle(
            self.mesh * x + self.half_mesh, self.mesh * y + self.half_mesh,
            self.mesh * (x+1) + self.half_mesh, self.mesh * (y+1) + self.half_mesh,
            fill=self.c_board_color, outline=self.c_board_color
        )
        # four corners and sides need attention
        a, b = [0, 1] if y == 0 else [-1, 0] if y == self.column-1 else [-1, 1]
        c, d = [0, 1] if x == 0 else [-1, 0] if x == self.row-1 else [-1, 1]
        self.c_gomoku.create_line(
            self.mesh * (x+1), self.mesh * (y+1) + a * self.half_mesh,
            self.mesh * (x+1), self.mesh * (y+1) + b * self.half_mesh,
            fill='black'
        )
        self.c_gomoku.create_line(
            self.mesh * (x+1) + c * self.half_mesh, self.mesh * (y+1),
            self.mesh * (x+1) + d * self.half_mesh, self.mesh * (y+1),
            fill='black'
        )

    def _draw_piece(self, x, y, tag:bool):
        '''draw one piece'''
        color = self._ternary_op('black', 'white', tag)
        self.c_gomoku.create_oval(
            self.mesh * (x+1) - self.piece_r, self.mesh * (y+1) + self.piece_r,
            self.mesh * (x+1) + self.piece_r, self.mesh * (y+1) - self.piece_r,
            fill=color
        )

    def _c_click(self, e):
        '''mouse trigger event'''
        if self.is_start == False or self.human == False: 
            return 

        x, y = int((e.x - self.half_mesh) / self.mesh), int((e.y - self.half_mesh) / self.mesh)
        if self.board[x][y] != 0:
            return

        self.last_x, self.last_y = x, y 

        self._draw_piece(x, y, self.is_black)
        self.board[x][y] = self._ternary_op(1, -1, self.is_black)

        self._gomoku_who_win()

        self.is_black = not self.is_black
        self.l_info.config(text=self._ternary_op('黑方行棋', '白方行棋', self.is_black))
        
        # self._Human_player() # just for testing
        # self._AI_player()
        self._ai_thread()

    def _ternary_op(self, black, white, tag:bool):
        '''ternary operator on whether the player is black or white'''
        return black if tag else white

    def _Human_player(self):
        '''human vs Human'''
        self.is_black = not self.is_black

    def _AI_player(self):
        '''the interface for AI
        Parameters required and updated: board status, which side to play 
        Return: the next gomoku piece coordinate (x, y)

        Gomoku Board status: 0 means no pieces, 1 means black pieces and -1 means white pieces
        '''

        self.human = False

        if self.is_start == False:
            return

        # AI_program

        AI = MCTS()
        [x, y] = AI.play(self.row, self.column, self.board)

        self._draw_piece(x, y, self.is_black)
        self.board[x][y] = self._ternary_op(1, -1, self.is_black)

        self.last_x, self.last_y = x, y 
        self._gomoku_who_win()

        self.is_black = not self.is_black
        self.l_info.config(text=self._ternary_op('黑方行棋', '白方行棋', self.is_black))
        self.human = True

    def _gomoku_who_win(self):
        '''Nothing to say, just brute force search is over'''
        end, winner = False, 0

        x, y = self.last_x, self.last_y
        four_dir = []
        four_dir.append([self.board[i][y] for i in range(self.row)])
        four_dir.append([self.board[x][j] for j in range(self.column)])
        def tilt_dir(x, y, dx, dy):
            cur = []
            while 0 <= x < self.row and 0 <= y < self.column:
                x, y = x + dx, y + dy
            x, y = x - dx, y - dy
            while 0 <= x < self.row and 0 <= y < self.column:
                cur.append(self.board[x][y])
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
                        end, winner = True, self._ternary_op(1, -1, self.is_black)
                else:
                    cnt = 0
        if end == False:
            return
        elif end == True and winner == 1:
            self._state_shift(2)
            showinfo("提示", "黑方获胜")
        elif end == True and winner == -1:
            self._state_shift(2)
            showinfo("提示", "白方获胜")
        else:
            pass

    # TODO: Go decision function
    def _go_who_win(self):
        pass

    def _ai_thread(self):
        '''Calling AI program takes a long time, open another thread for execution.'''
        self.thread = threading.Thread(target=self._AI_player)
        self.thread.start()

if __name__ == '__main__':
    gomoku = Gomoku(10, 10)
    gomoku.run()


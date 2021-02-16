from tkinter import * 
from tkinter.messagebox import *

class Gomoku():
    def __init__(self, row=8, column=8):
        self.row, self.column = row, column
        self.mesh = 25
        self.half_mesh = self.mesh / 2
        self.factor = 0.9
        
        self.f_header_bg = "#e5e1ed"
        self.c_board_color = "#e6b953"
        self.b_font_info = ("楷体", 12, "bold")

    def run(self):
        self.root = Tk()
        self.root.title("Gomoku")
        self.root.resizable(height = None, width = None)

        self.f_header = Frame(self.root, bg=self.f_header_bg)
        self.f_header.pack(fill=BOTH, ipadx=100)

        self.b_start = Button(self.f_header, text="开始", command=self._start, font=self.b_font_info)
        self.b_restart = Button(self.f_header, text="重开", command=self._restart, font=self.b_font_info)
        self.l_info = Label(self.f_header, text="准备开始", bg=self.f_header_bg, font=("楷体", 14, "bold"), fg="grey")
        self.b_regret = Button(self.f_header, text="悔棋", command=self._regret, font=self.b_font_info)
        self.b_defeat = Button(self.f_header, text="认输", command=self._defeat, font=self.b_font_info)

        self.b_start.pack(side=LEFT, padx=20)
        self.b_restart.pack(side=LEFT)
        self.l_info.pack(side=LEFT, expand=YES, fill=BOTH)
        self.b_regret.pack(side=RIGHT)
        self.b_defeat.pack(side=RIGHT, padx=20)

        self.c_gomoku = Canvas(self.root, bg=self.c_board_color, width=(self.column+1)*self.mesh, height=(self.row+1)*self.mesh)

        #1 TODO: board
        #2 TODO: click event
        self.c_gomoku.pack()

        self.root.mainloop()

    def _start(self):
        pass

    def _restart(self):
        pass

    def _regret(self):
        pass

    def _defeat(self):
        pass

if __name__ == '__main__':
    gomoku = Gomoku()
    gomoku.run()
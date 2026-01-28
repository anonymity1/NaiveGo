from tkinter import *
from tkinter.messagebox import *
import threading
import math

# 尝试导入 AI 模块，没有则忽略
try:
    from alpha import Alpha
except ImportError:
    pass

class Gomoku:
    def __init__(self, rows=15, cols=15):
        self.rows = rows
        self.cols = cols
        
        # --- 视觉风格配置 (Katago/Sabaki 风格) ---
        self.theme = {
            'board_bg': "#E3C686",       # 榧木色 (Kaya Wood)
            'line_color': "#5E4826",     # 深褐色线条，比纯黑更柔和
            'coord_font': "Helvetica",   # 坐标字体
            'last_move_color': "#FF4500",# 最新落子标记颜色 (红橙色)
            
            # 黑棋风格
            'black_body': "#1A1A1A",     # 墨黑 (非纯黑)
            'black_outline': "#000000",
            'black_highlight': "#444444",# 哑光高光
            
            # 白棋风格
            'white_body': "#FDFDFD",     # 贝壳白
            'white_outline': "#B0B0B0",  # 灰色边缘防锯齿
            'white_shadow': "#DCDCDC",   # 内部阴影
            
            # 阴影 (模拟悬浮感)
            'shadow_color': "#8A6E40",   # 投射在棋盘上的阴影
        }
        
        # 游戏状态
        self.board = [[0] * self.cols for _ in range(self.rows)]
        self.last_move = None
        self.is_black = True
        self.game_state = 'IDLE' 
        self.is_human_turn = False
        
        # 动态布局参数 (会在 resize 时自动更新)
        self.mesh = 30
        self.margin = 30
        self.stone_r = 13
        self.offset_y = 30  # <--- 【添加这一行】 给它一个默认值

    def run(self, model_file=None):
        self.model_file = model_file
        self.root = Tk()
        self.root.title(f"Gomoku AI (Katago Style) - {self.cols}x{self.rows}")
        
        # 允许窗口调整大小
        self.root.geometry("800x850") 
        self.root.minsize(500, 550)

        self._init_widgets()
        self._reset_game()
        
        self.root.mainloop()

    def _init_widgets(self):
        # 1. 顶部控制栏
        f_header = Frame(self.root, bg="#f0f0f0", pady=10)
        f_header.pack(side=TOP, fill=X)

        btn_style = {"font": ("微软雅黑", 11), "padx": 10, "bg": "white", "relief": "groove"}
        
        self.btn_black = Button(f_header, text="执黑(先手)", command=lambda: self._start_game(True), **btn_style)
        self.btn_white = Button(f_header, text="执白(后手)", command=lambda: self._start_game(False), **btn_style)
        self.btn_restart = Button(f_header, text="重开一局", command=self._reset_game, state=DISABLED, **btn_style)
        self.l_info = Label(f_header, text="准备开始", font=("微软雅黑", 14, "bold"), bg="#f0f0f0", fg="#333")

        self.btn_black.pack(side=LEFT, padx=15)
        self.btn_white.pack(side=LEFT, padx=5)
        self.l_info.pack(side=LEFT, expand=YES)
        self.btn_restart.pack(side=RIGHT, padx=15)

        # 2. 棋盘画布 (关键：pack fill=BOTH expand=YES 以支持缩放)
        self.canvas = Canvas(self.root, bg=self.theme['board_bg'], highlightthickness=0)
        self.canvas.pack(side=BOTTOM, fill=BOTH, expand=YES)
        
        # 绑定事件
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", self._on_resize) # 监听窗口大小变化

    def _on_resize(self, event):
        """窗口大小改变时，重新计算网格并重绘"""
        w, h = event.width, event.height
        
        # 计算网格大小：取宽高的较小值，留出边距
        # padding_ratio 控制棋盘边缘留白
        padding_ratio = 0.9 
        min_side = min(w, h)
        
        # 最大的行或列数
        max_grid = max(self.rows, self.cols)
        
        # 动态计算 mesh
        self.mesh = (min_side * padding_ratio) / (max_grid + 1)
        self.margin = (w - self.mesh * (self.cols - 1)) / 2 if w > h else self.mesh * 1.5
        # 垂直居中偏移
        self.offset_y = (h - self.mesh * (self.rows - 1)) / 2
        
        # 棋子半径 (占网格的 45%)
        self.stone_r = self.mesh * 0.45
        
        self._redraw_all()

    def _redraw_all(self):
        """重绘整个界面（棋盘+棋子）"""
        self.canvas.delete("all")
        self._draw_board_grid()
        self._draw_coordinates()
        self._draw_existing_stones()

    def _draw_board_grid(self):
        """绘制棋盘线和星位"""
        # 1. 画线
        for r in range(self.rows):
            y = self.offset_y + r * self.mesh
            start_x = self.margin
            end_x = self.margin + (self.cols - 1) * self.mesh
            self.canvas.create_line(start_x, y, end_x, y, fill=self.theme['line_color'], width=1.5)

        for c in range(self.cols):
            x = self.margin + c * self.mesh
            start_y = self.offset_y
            end_y = self.offset_y + (self.rows - 1) * self.mesh
            self.canvas.create_line(x, start_y, x, end_y, fill=self.theme['line_color'], width=1.5)

        # 2. 画星位 (自适应不同棋盘大小)
        if self.rows == 19 and self.cols == 19:
            points = [(3,3), (3,9), (3,15), (9,3), (9,9), (9,15), (15,3), (15,9), (15,15)]
        elif self.rows == 15 and self.cols == 15:
            points = [(3,3), (3,11), (7,7), (11,3), (11,11)]
        else:
            points = [] # 其他尺寸暂不画星位

        r_star = self.mesh * 0.12 # 星位点半径
        for r, c in points:
            cx = self.margin + c * self.mesh
            cy = self.offset_y + r * self.mesh
            self.canvas.create_oval(cx-r_star, cy-r_star, cx+r_star, cy+r_star, fill=self.theme['line_color'])

    def _draw_coordinates(self):
        """绘制坐标数字（支持缩放）"""
        font_size = int(self.mesh * 0.4)
        if font_size < 8: return # 太小就不画了
        
        font = (self.theme['coord_font'], font_size)
        
        # 画列号 (数字 1, 2, 3...)
        for c in range(self.cols):
            x = self.margin + c * self.mesh
            y = self.offset_y - self.mesh * 0.8
            self.canvas.create_text(x, y, text=str(c+1), font=font, fill=self.theme['line_color'])

        # 画行号 (字母 A, B, C...)
        for r in range(self.rows):
            x = self.margin - self.mesh * 0.8
            y = self.offset_y + r * self.mesh
            txt = chr(65 + r) # A, B, C...
            self.canvas.create_text(x, y, text=txt, font=font, fill=self.theme['line_color'])

    def _draw_stone_body(self, x, y, is_black, tag_id=None):
        """
        绘制拟物化棋子
        原理：阴影 -> 棋子本体 -> 内部高光/光泽
        """
        r = self.stone_r
        
        # 1. 投射阴影 (Shadow) - 稍微向右下偏移
        shadow_offset = r * 0.15
        self.canvas.create_oval(
            x - r + shadow_offset, y - r + shadow_offset,
            x + r + shadow_offset, y + r + shadow_offset,
            fill=self.theme['shadow_color'], outline="", tags=tag_id
        )

        if is_black:
            # 2. 黑棋本体 (Matte Black)
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=self.theme['black_body'], outline=self.theme['black_outline'], 
                tags=tag_id
            )
            # 3. 黑棋高光 (右上角微弱反光)
            hl_r = r * 0.5
            hl_offset_x = -r * 0.3
            hl_offset_y = -r * 0.3
            self.canvas.create_oval(
                x + hl_offset_x - hl_r/2, y + hl_offset_y - hl_r/2,
                x + hl_offset_x + hl_r/2, y + hl_offset_y + hl_r/2,
                fill=self.theme['black_highlight'], outline="", 
                tags=tag_id
            )
        else:
            # 2. 白棋本体 (Shell White)
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=self.theme['white_body'], outline=self.theme['white_outline'],
                tags=tag_id
            )
            # 3. 白棋质感 (模拟贝壳纹理，这里用简单的内部光泽代替)
            # 在内部画一个稍小的圆，颜色稍深，模拟弧度
            inner_r = r * 0.85
            self.canvas.create_oval(
                x - inner_r, y - inner_r, x + inner_r, y + inner_r,
                fill="", outline=self.theme['white_shadow'], width=1,
                tags=tag_id
            )

    def _draw_existing_stones(self):
        """重绘所有已存在的棋子"""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] != 0:
                    x = self.margin + c * self.mesh
                    y = self.offset_y + r * self.mesh
                    self._draw_stone_body(x, y, self.board[r][c] == 1)
        
        # 重新标记最后一步
        if self.last_move:
            self._mark_last_move(self.last_move[0], self.last_move[1])

    def _draw_piece_animate(self, r, c, is_black):
        """落子逻辑"""
        x = self.margin + c * self.mesh
        y = self.offset_y + r * self.mesh
        self._draw_stone_body(x, y, is_black)
        self._mark_last_move(r, c)

    def _mark_last_move(self, r, c):
        """在最后一步的棋子上画标记"""
        # 先清除旧标记
        self.canvas.delete("last_move_marker")
        
        x = self.margin + c * self.mesh
        y = self.offset_y + r * self.mesh
        
        # 标记风格：根据棋子颜色选择标记颜色
        # 黑棋用红色三角形，白棋用红色三角形（或者其他显眼记号）
        marker_r = self.stone_r * 0.3
        
        # 简单的三角形标记
        self.canvas.create_polygon(
            x, y - marker_r,
            x - marker_r, y + marker_r * 0.6,
            x + marker_r, y + marker_r * 0.6,
            fill=self.theme['last_move_color'], outline="",
            tags="last_move_marker"
        )
        
        # 或者显示手数数字 (Katago风格)
        # step_num = ... 需要维护步数变量
        # self.canvas.create_text(x, y, text=str(step_num), fill="red"...)

    def _on_canvas_click(self, event):
        if self.game_state != 'PLAYING' or not self.is_human_turn:
            return

        # 鼠标点击坐标转换：(mx - margin) / mesh
        # 使用 round 实现吸附
        try:
            c = round((event.x - self.margin) / self.mesh)
            r = round((event.y - self.offset_y) / self.mesh)
        except ZeroDivisionError:
            return

        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
        if self.board[r][c] != 0:
            return

        self._perform_move(r, c)

    def _perform_move(self, r, c):
        val = 1 if self.is_black else -1
        self.board[r][c] = val
        self.last_move = (r, c)
        
        self._draw_piece_animate(r, c, self.is_black)
        
        if self._check_win(r, c):
            winner = "黑方" if self.is_black else "白方"
            self.game_state = 'END'
            self.l_info.config(text=f"{winner}获胜!", fg="red")
            showinfo("游戏结束", f"{winner}获胜!")
            return

        self.is_black = not self.is_black
        self.is_human_turn = not self.is_human_turn
        self._update_info_label()

        if not self.is_human_turn and self.game_state == 'PLAYING':
            self._start_ai_thread()

    def _update_info_label(self):
        color_text = "黑方" if self.is_black else "白方"
        self.l_info.config(text=f"当前回合: {color_text}", fg="#333")

    def _check_win(self, r, c):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        color = self.board[r][c]
        for dr, dc in directions:
            count = 1
            tr, tc = r + dr, c + dc
            while 0 <= tr < self.rows and 0 <= tc < self.cols and self.board[tr][tc] == color:
                count += 1
                tr += dr
                tc += dc
            tr, tc = r - dr, c - dc
            while 0 <= tr < self.rows and 0 <= tc < self.cols and self.board[tr][tc] == color:
                count += 1
                tr -= dr
                tc -= dc
            if count >= 5: return True
        return False

    def _reset_game(self):
        self.game_state = 'IDLE'
        self.board = [[0] * self.cols for _ in range(self.rows)]
        self.last_move = None
        self._redraw_all()
        
        self.btn_black.config(state=NORMAL)
        self.btn_white.config(state=NORMAL)
        self.btn_restart.config(state=DISABLED)
        self.l_info.config(text="准备开始", fg="#333")

    def _start_game(self, start_as_black):
        self.game_state = 'PLAYING'
        self.is_black = True
        self.is_human_turn = start_as_black
        
        self.btn_black.config(state=DISABLED)
        self.btn_white.config(state=DISABLED)
        self.btn_restart.config(state=NORMAL)
        self._update_info_label()

        if not self.is_human_turn:
            self._start_ai_thread()

    def _start_ai_thread(self):
        thread = threading.Thread(target=self._ai_logic)
        thread.daemon = True
        thread.start()

    def _ai_logic(self):
        try:
            # ------------------------
            # 在这里接入你的 AI
            AI = Alpha(model_file=self.model_file)
            move = AI.play(self.rows, self.cols, self.board)
            # 假设返回 move = [row, col] 或 [col, row]，请根据你的AI实际返回值确认
            # ------------------------
            # ------------------------
            # import time
            # time.sleep(0.3) 
            
            # # 模拟 AI 下棋逻辑
            # move = None
            # # 简单的策略：找中间的空位
            # center_r, center_c = self.rows // 2, self.cols // 2
            # if self.board[center_r][center_c] == 0:
            #     move = [center_r, center_c]
            # else:
            #     for r in range(self.rows):
            #         for c in range(self.cols):
            #             if self.board[r][c] == 0:
            #                 move = [r, c]
            #                 break
            #         if move: break
            
            if move:
                self.root.after(0, lambda: self._perform_move(move[0], move[1]))
                
        except Exception as e:
            print(f"AI Error: {e}")

if __name__ == '__main__':
    # 启动 15x15 棋盘
    game = Gomoku(rows=8, cols=8)
    game.run(model_file='./best_model/best_model_8x8')
import numpy as np
import time
import sys
import tkinter as tk

Unit = 60
Maze_h = 6
Maze_w = 6


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title("maze")
        self.geometry('{0}x{1}'.format(Maze_h * Unit, Maze_w * Unit))
        self.n_features = 4
        self._build_maze()




    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=Maze_h * Unit,
                                width=Maze_w * Unit)

        for c in range(0, Maze_w * Unit, Unit):
            x0, y0, x1, y1 = c, 0, c, Maze_h * Unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, Maze_h * Unit, Unit):
            x0, y0, x1, y1 = 0, r, Maze_w * Unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([30, 30])

        hell1_center = origin + np.array([Unit * 2, Unit])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 20, hell1_center[1] - 20,
            hell1_center[0] + 20, hell1_center[1] + 20,
            fill='black')

        hell2_center = origin + np.array([Unit, Unit * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 20, hell2_center[1] - 20,
            hell2_center[0] + 20, hell2_center[1] + 20,
            fill='black')

        hell3_center = origin + np.array([Unit * 2, Unit * 2])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 20, hell3_center[1] - 20,
            hell3_center[0] + 20, hell3_center[1] + 20,
            fill='black')

        hell4_center = origin + np.array([Unit * 5, Unit * 2])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 20, hell4_center[1] - 20,
            hell4_center[0] + 20, hell4_center[1] + 20,
            fill='black')

        hell5_center = origin + np.array([Unit * 2, Unit * 5])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 20, hell5_center[1] - 20,
            hell5_center[0] + 20, hell5_center[1] + 20,
            fill='black')
        # hell6_center = origin + np.array([Unit * 3, Unit * 4])
        # self.hell6 = self.canvas.create_rectangle(
        #     hell6_center[0] - 20, hell6_center[1] - 20,
        #     hell6_center[0] + 20, hell6_center[1] + 20,
        #     fill='black')
        # hell7_center = origin + np.array([Unit *4, Unit*0])
        # self.hell7 = self.canvas.create_rectangle(
        #     hell7_center[0] - 20, hell7_center[1] - 20,
        #     hell7_center[0] + 20, hell7_center[1] + 20,
        #     fill='black')
        # hell8_center = origin + np.array([Unit * 0, Unit * 4])
        # self.hell8 = self.canvas.create_rectangle(
        #     hell8_center[0] - 20, hell8_center[1] - 20,
        #     hell8_center[0] + 20, hell8_center[1] + 20,
        #     fill='black')


        oval_center = origin + np.array([Unit * 3, Unit * 5])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 20, oval_center[1] - 20,
            oval_center[0] + 20, oval_center[1] + 20,
            fill='yellow')

        self.rect = self.canvas.create_rectangle(
            origin[0] - 20, origin[1] - 20,
            origin[0] + 20, origin[1] + 20,
            fill='red')

        self.canvas.pack()


    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([30, 30])
        self.rect = self.canvas.create_rectangle(
            origin[0] -20, origin[1] -20,
            origin[0] +20, origin[1] +20,
            fill='red')
        return self.canvas.coords(self.rect)


    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:
            if s[1] > Unit:
                base_action[1] -= Unit
        elif action == 1:
            if s[1] < (Maze_h - 1) * Unit:
                base_action[1] += Unit
        elif action == 2:
            if s[0] > Unit:
                base_action[0] -= Unit
        elif action == 3:
            if s[0] < (Maze_w - 1) * Unit:
                base_action[0] += Unit

        self.canvas.move(self.rect, base_action[0], base_action[1])

        s_ = self.canvas.coords(self.rect)

        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            # s_ = self.reset()

        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3),
                    self.canvas.coords(self.hell4),self.canvas.coords(self.hell5)]:
            reward = -1
            done = True
            # s_ = self.reset()

        else:
            reward = 0
            done = False
            # s_ = self.reset()

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s,r,done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()





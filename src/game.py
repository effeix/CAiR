from random import randint
import numpy as np
import matplotlib.pyplot as plt


class Game(object):

    bd_colors = {
        "BLACK": 0,
        "RED": 1,
        "GREEN": 2,
        "BLUE": 3
    }

    im_colors = {
        "BLACK": np.array([0, 0, 0]),
        "RED": np.array([255, 0, 0]),
        "GREEN": np.array([0, 255, 0]),
        "BLUE": np.array([0, 0, 255])
    }

    def __init__(self, size=8, n_green=3, n_red=3):

        self.size = size
        self.orig_n_green = n_green
        self.orig_n_red = n_red
        self.reset()

    def reset(self):
        self.color_hit_after_move = "BLACK"
        self.last_player_pos = [0, 0]
        self.n_green = self.orig_n_green
        self.n_red = self.orig_n_red
        self.player_pos = [0, 0]
        self.game_state = "neutral"
        self.tiles_info = {}

        self.bd = np.zeros((self.size, self.size), dtype=np.uint8)
        self.im = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        self.__set_color(self.player_pos, "BLUE")

        self.__generate_random_tiles()

        return self.im

    def __set_color(self, p, c):
        self.bd[p[0]][p[1]] = Game.bd_colors[c]
        self.im[p[0]][p[1]] = Game.im_colors[c]

    def __get_color(self, p, typ="bd"):
        if typ == "bd":
            return self.bd[p[0]][p[1]]

        elif typ == "im":
            return self.im[p[0]][p[1]]

    def __hit_border(self, border):

        hit = False

        if border == 0:
            if self.player_pos[0] - 1 < 0:
                hit = True

        elif border == 1:
            if self.player_pos[1] + 1 > self.size - 1:
                hit = True

        elif border == 2:
            if self.player_pos[0] + 1 > self.size - 1:
                hit = True

        elif border == 3:
            if self.player_pos[1] - 1 < 0:
                hit = True

        return hit

    def move_player(self, direction):

        moved = False

        if not self.__hit_border(direction):

            self.last_player_pos = self.player_pos[:]

            if direction == 0:
                self.player_pos[0] -= 1

            elif direction == 1:
                self.player_pos[1] += 1

            elif direction == 2:
                self.player_pos[0] += 1

            elif direction == 3:
                self.player_pos[1] -= 1

            moved = True
            self.__update_game()

        return moved

    def __update_game(self):
        self.color_hit_after_move = self.__get_color(self.player_pos)

        self.__set_color(self.last_player_pos, "BLACK")
        self.__set_color(self.player_pos, "BLUE")

        if self.color_hit_after_move == 0:
            self.game_state = "neutral"

        elif self.color_hit_after_move == 1:
            self.game_state = "lost"

        elif self.color_hit_after_move == 2:
            self.game_state = "collected"

            self.n_green -= 1

            print("n_green", self.n_green)
            if self.n_green == 0:
                self.game_state = "won"

    def get_game_state(self):
        return self.game_state

    def get_image(self):
        return self.im

    def display(self):
        plt.imshow(self.im)
        plt.show()

    def __generate_random_tiles(self):

        print("Generating...")

        # Available positions for tile placement
        # Third element represents color:
        #   0 (black), 1 (red), 2 (green), 3 (blue)
        avail_pos = []

        for r in range(self.size):
            for c in range(self.size):
                avail_pos.append([r, c])

        avail_pos.remove([0, 0])

        # Generate green tiles
        for _ in range(self.n_green):
            while True:
                pos = [randint(0, self.size - 1), randint(0, self.size - 1)]

                if pos in avail_pos:
                    break

                pos = [randint(0, self.size - 1), randint(0, self.size - 1)]

            self.__set_color(pos, "GREEN")
            avail_pos.remove(pos)

        # Red tiles cannot be next to player on start,
        # so block right and bottom if they are available
        try:
            avail_pos.remove([0, 1])
        except ValueError:
            pass

        try:
            avail_pos.remove([1, 0])
        except ValueError:
            pass

        # Generate red tiles
        for _ in range(self.n_red):
            while True:
                pos = [randint(0, self.size - 1), randint(0, self.size - 1)]

                if pos in avail_pos:
                    break

                pos = [randint(0, self.size - 1), randint(0, self.size - 1)]

            self.__set_color(pos, "RED")
            avail_pos.remove(pos)

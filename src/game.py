from random import choice
import numpy as np


class Game(object):

    STATE_NEUTRAL = 0
    STATE_COLLECTED = 1
    STATE_WON = 2
    STATE_LOST = 3

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

    def __init__(self, size=16, n_green=3, n_red=3):

        self.size = size
        self.orig_n_green = n_green
        self.orig_n_red = n_red

    def reset(self):
        self.__is_first_display = True

        self.color_hit_after_move = "BLACK"

        self.n_green = self.orig_n_green
        self.n_red = self.orig_n_red

        self.last_player_pos = [0, 0]
        self.player_pos = [0, 0]

        self.game_state = Game.STATE_NEUTRAL

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

    def get_game_state(self):
        return self.game_state

    def get_image(self):
        return self.im

    def get_board(self):
        return self.bd

    def __generate_random_tiles(self):

        avail_pos = [
            pos
            for pos in np.ndindex(self.size, self.size)
            if pos != (0, 0)
        ]

        for idx in range(-self.n_green, self.n_red):
            color = "GREEN" if idx < 0 else "RED"

            pos = avail_pos.pop(avail_pos.index(choice(avail_pos)))

            self.__set_color(pos, color)

    def __update_game(self):
        self.color_hit_after_move = self.__get_color(self.player_pos, typ="im")
        print("Color hit:", self.color_hit_after_move)

        if self.color_hit_after_move == 0 or self.color_hit_after_move == 3:
            self.game_state = Game.STATE_NEUTRAL

        elif self.color_hit_after_move == 1:
            self.game_state = Game.STATE_LOST

        elif self.color_hit_after_move == 2:
            self.game_state = Game.STATE_COLLECTED

            self.n_green -= 1

            if self.n_green == 0:
                self.game_state = Game.STATE_WON

        self.__set_color(self.last_player_pos, "BLACK")
        self.__set_color(self.player_pos, "BLUE")

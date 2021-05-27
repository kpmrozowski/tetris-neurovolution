"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
import random

style.use("ggplot")


class Tetris:
    block_size = 30
    piece_colors = [
        np.ones((block_size, block_size, 3), dtype=np.uint8) * np.array([0, 0, 0]),
        np.ones((block_size, block_size, 3), dtype=np.uint8) * np.array([255, 255, 0]),
        np.ones((block_size, block_size, 3), dtype=np.uint8) * np.array([147, 88, 254]),
        np.ones((block_size, block_size, 3), dtype=np.uint8) * np.array([54, 175, 144]),
        np.ones((block_size, block_size, 3), dtype=np.uint8) * np.array([255, 0, 0]),
        np.ones((block_size, block_size, 3), dtype=np.uint8) * np.array([102, 217, 238]),
        np.ones((block_size, block_size, 3), dtype=np.uint8) * np.array([254, 151, 32]),
        np.ones((block_size, block_size, 3), dtype=np.uint8) * np.array([0, 0, 255])
    ]

    pieces = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5, 5, 5, 5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]]
    ]

    score = 0

    def __init__(self, height=20, width=10, block_size=20):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.extra_board = np.ones(((3 + self.height) * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
        self.text_color = (200, 20, 220)
        self.collision = False
        self.reset()

    def reset(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.ind_next = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.score = 0
        self.gameover = False
        return self.get_state_properties(self.board)

    def rotate(self, piece):
        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array

    def get_state_properties(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def get_holes(self, board):
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board):
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_next_states(self):
        states = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]
        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_state_properties(board)
            curr_piece = self.rotate(curr_piece)
        return states

    def get_current_board_state(self):
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        return board

    def new_piece(self):
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.ind_next
        self.ind_next = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                            "y": 0
                            }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    def truncate(self, piece, pos):
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    def store(self, piece, pos):
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

    def step(self, action, render=True, video=None):
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
        if render:
            self.render(video)

        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True

        self.board = self.store(self.piece, self.current_pos)

        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        self.tetrominoes += 1
        self.cleared_lines += lines_cleared
        if not self.gameover:
            self.new_piece()
        if self.gameover:
            self.score -= 2

        return score, self.gameover

    def render(self, video=None):
        img = np.zeros((self.height * self.block_size, self.width * self.block_size, 3))
        if not self.gameover:
            ii = 0
            for row in self.get_current_board_state():
                jj = 0
                for p in row:
                    block = self.piece_colors[p]
                    img[ii*self.block_size:(ii + 1)*self.block_size, jj*self.block_size:(jj + 1)*self.block_size] = block
                    jj += 1
                ii += 1
        else:
            ii = 0
            for row in self.board:
                jj = 0
                for p in row:
                    block = self.piece_colors[p]
                    img[ii*self.block_size:(ii + 1)*self.block_size, jj*self.block_size:(jj + 1)*self.block_size] = block
                    jj += 1
                ii += 1

        piece_i = [row[:] for row in self.pieces[self.ind_next]]
        piece_rows = len(piece_i)
        piece_cols = len(piece_i[0])
        next_piece_1 = np.zeros((piece_rows * self.block_size, piece_cols * self.block_size, 3))
        ii = 0
        for row in self.pieces[self.ind_next]:
            jj = 0
            for p in row:
                next_piece_1[ii*self.block_size:(ii + 1)*self.block_size, jj*self.block_size:(jj + 1)*self.block_size] = self.piece_colors[p]
                jj += 1
            ii += 1

        next_piece_1[[i * self.block_size for i in range(piece_rows)], :, :] = 0
        next_piece_1[:, [i * self.block_size for i in range(piece_cols)], :] = 0
        next_piece_1 = np.pad(next_piece_1, [(0, (2 - piece_rows) * self.block_size), ((4 - piece_cols) * self.block_size, 0), (0, 0)])

        piece_left_padding = np.ones(((self.width - 4) * self.block_size, 2 * self.block_size, 3),
                                   dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
        piece_left_padding = Image.fromarray(piece_left_padding, "RGB")
        piece_left_padding = piece_left_padding.resize(((self.width - 4) * self.block_size, 2 * self.block_size))

        next_piece_2 = np.concatenate((piece_left_padding, next_piece_1), axis=1)

        piece_bottom_padding = np.ones((self.width * self.block_size, self.block_size, 3),
                                   dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
        piece_bottom_padding = Image.fromarray(piece_bottom_padding, "RGB")
        piece_bottom_padding = piece_bottom_padding.resize((self.width * self.block_size, self.block_size))

        next_piece_img = np.concatenate((next_piece_2, piece_bottom_padding), axis=0)

        img[[i * self.block_size for i in range(self.height)], :, :] = 0
        img[:, [i * self.block_size for i in range(self.width)], :] = 0

        img = np.concatenate((next_piece_img, img), axis=0)

        img = np.concatenate((img, self.extra_board), axis=1)
        img = np.array(img, dtype=np.uint8)

        cv2.putText(img, "Next " + str(self.ind_next), (self.width * self.block_size + int(self.block_size / 2), self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.score),
                    (self.width * self.block_size + int(self.block_size / 2), 6 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.tetrominoes),
                    (self.width * self.block_size + int(self.block_size / 2), 9 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 11 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.cleared_lines),
                    (self.width * self.block_size + int(self.block_size / 2), 12 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        # if video:
        #     video.write(img)
        cv2.imshow("Deep Q-Learning Tetris", img)
        cv2.waitKey(1)

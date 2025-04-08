import numpy as np
from enum import Enum


class Env(Enum):
    EMPTY = 0
    WALL = 1
    APPLE_RED = 2
    APPLE_GREEN = 3
    SNAKE_HEAD = 4
    SNAKE_BODY = 5


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    IDLE = 4


class GameState(Enum):
    OK = 0
    GAMEOVER = 1


class SnakeGame:
    def __init__(self, w: int, h: int, red: int = 2, green: int = 1) -> None:
        self.env = np.ndarray = None
        self._buffer = np.ndarray = None
        self.snake = np.ndarray = None
        self._points: int = 0
        self.reset(w, h)

    def _get_random_coords(self) -> np.ndarray:
        return np.random.randint(2, [self.env.shape[0], self.env.shape[1]])

    def _set_buffer(self, buffer: np.ndarray, coords: np.ndarray, value: Env) -> None:
        buffer[coords[0], coords[1]] = value.value

    def _get_buffer(self, buffer: np.ndarray, coords: np.ndarray) -> int:
        return buffer[coords[0], coords[1]]

    def _get_empty_coords(self, buffer: np.ndarray) -> np.ndarray:
        coords: np.ndarray = self._get_random_coords()
        while self._get_buffer(buffer, coords) != Env.EMPTY.value:
            coords = self._get_random_coords()
        return coords

    def _spawn_obj(self, obj: Env, buffer: np.ndarray) -> None:
        coords: np.ndarray = self._get_empty_coords(buffer)
        self._set_buffer(buffer, coords, obj)

    def _update_snake(self, buffer: np.ndarray) -> None:
        self._set_buffer(buffer, self.snake[0], Env.SNAKE_HEAD)
        for segment in self.snake[1:]:
            self._set_buffer(buffer, segment, Env.SNAKE_BODY)

    def _is_valid_coord(self, buffer: np.ndarray, coords: np.ndarray) -> None:
        if coords[0] < 0 or coords[0] >= buffer.shape[0]:
            return False
        if coords[1] < 0 or coords[1] >= buffer.shape[1]:
            return False
        if self._get_buffer(buffer, coords) != Env.EMPTY.value:
            return False
        if np.any(np.all(self.snake == np.array([coords[0], coords[1]]), axis=1)):
            return False
        return True

    def _grow_snake(self, buffer: np.ndarray) -> None:
        next = np.array(self.snake[-1])
        if self._is_valid_coord(buffer, next + np.array([1, 0])):
            next[0] += 1
        elif self._is_valid_coord(buffer, next + np.array([-1, 0])):
            next[0] -= 1
        elif self._is_valid_coord(buffer, next + np.array([0, 1])):
            next[1] += 1
        elif self._is_valid_coord(buffer, next + np.array([0, -1])):
            next[1] -= 1
        self.snake = np.append(self.snake, [next], axis=0)

    def _update_snake_body(self) -> None:
        for i in range(len(self.snake) - 1):
            idx: int = -(i + 1)
            self.snake[idx] = self.snake[idx - 1]

    def _update_back_buffer(self) -> None:
        for x in range(self.env.shape[0]):
            for y in range(self.env.shape[1]):
                if (
                    self.env[x, y] == Env.APPLE_GREEN.value
                    or self.env[x, y] == Env.APPLE_RED.value
                ):
                    self._buffer[x, y] = self.env[x, y]

    def reset(
        self, w: int, h: int, red: int = 2, green: int = 1, length: int = 2
    ) -> None:
        self.points = 0
        self.env = np.zeros((w, h), dtype=int)
        self._buffer = np.zeros(self.env.shape, dtype=int)
        for _ in range(red):
            self._spawn_obj(Env.APPLE_RED, self.env)
        for _ in range(green):
            self._spawn_obj(Env.APPLE_GREEN, self.env)
        self.snake = np.empty((1, 2), dtype=int)
        self.snake[0] = self._get_empty_coords(self.env)
        for _ in range(length):
            self._grow_snake(self.env)
        self._update_snake(self.env)

    def update(self, action: Action) -> GameState:
        if action == Action.IDLE:
            return GameState.OK
        state: GameState = GameState.OK
        self._update_snake_body()
        if action == action.UP:
            self.snake[0] += np.array([0, -1])
        elif action == action.DOWN:
            self.snake[0] += np.array([0, 1])
        elif action == action.LEFT:
            self.snake[0] += np.array([-1, 0])
        elif action == action.RIGHT:
            self.snake[0] += np.array([1, 0])
        if self.snake[0][0] < 0 or self.snake[0][0] >= self.env.shape[0]:
            return GameState.GAMEOVER
        elif self.snake[0][1] < 0 or self.snake[0][1] >= self.env.shape[1]:
            return GameState.GAMEOVER
        self._buffer = np.zeros(self.env.shape, dtype=int)
        self._update_snake(self._buffer)
        self._update_back_buffer()

        if self._get_buffer(self.env, self.snake[0]) == Env.APPLE_GREEN.value:
            self._grow_snake(self.env)
            self._points += 1
            self._spawn_obj(Env.APPLE_GREEN, self._buffer)
        elif self._get_buffer(self.env, self.snake[0]) == Env.APPLE_RED.value:
            self._set_buffer(self._buffer, self.snake[-1], Env.EMPTY)
            self.snake = self.snake[:-1]
            self._points -= 1
            self._spawn_obj(Env.APPLE_RED, self._buffer)
        elif self._get_buffer(self.env, self.snake[0]) == Env.SNAKE_BODY.value:
            return GameState.GAMEOVER

        self._update_snake(self._buffer)
        self.env = self._buffer
        return state

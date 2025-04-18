import numpy as np
from defs import EnvID, QReward, Action


class Environment:
    def __init__(self, w: int, h: int) -> None:
        self._N_RED = 2
        self._N_GREEN = 1
        self._SNAKE_LEN = 2
        self.w = w
        self.h = h
        self._env: np.ndarray = None
        self.snake: np.ndarray = None
        self._dirs: np.ndarray = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        self.reset(w, h)

    def get_env(self) -> np.ndarray:
        return np.copy(self._env)

    def get_cell(self, x: int, y: int) -> int:
        return self._env[y, x]

    def _set_cell(self, x: int, y: int, id: EnvID) -> None:
        self._env[y, x] = id.value

    def _is_available_cell(self, x: int, y: int) -> bool:
        return self.get_cell(x, y) == EnvID.EMPTY.value

    def _get_available_cell(self) -> np.ndarray:
        cell: np.ndarray = np.zeros((2), dtype=int)
        while not self._is_available_cell(cell[0], cell[1]):
            cell[0] = np.random.randint(0, self.w)
            cell[1] = np.random.randint(0, self.h)
            # cell = np.random.randint(2, [self.w, self.h])
        return cell

    def _spawn_obj(self, id: EnvID) -> None:
        coords: np.ndarray = self._get_available_cell()
        self._set_cell(coords[0], coords[1], id)

    def _init_snake(self) -> None:
        self.snake = np.empty((1, 2), dtype=int)
        self.snake[0] = self._get_available_cell()

    def _is_in_snake(self, x: int, y: int) -> bool:
        target: np.ndarray = np.array([x, y])
        return np.any(np.all(self.snake == target, axis=1))

    def _extend_snake(self) -> None:
        np.random.shuffle(self._dirs)
        for dir in self._dirs:
            next: np.ndarray = self.snake[-1] + dir
            if self._is_available_cell(next[0], next[1]) and not self._is_in_snake(
                next[0], next[1]
            ):
                self.snake = np.append(self.snake, [next], axis=0)
                return

    def _snake_to_env(self) -> None:
        self._env = np.where(
            self._env == EnvID.SNAKE_BODY.value, EnvID.EMPTY.value, self._env
        )
        self._env = np.where(
            self._env == EnvID.SNAKE_HEAD.value, EnvID.EMPTY.value, self._env
        )
        if self.snake.shape[0] != 0:
            self._set_cell(self.snake[0][0], self.snake[0][1], EnvID.SNAKE_HEAD)
        for body_segment in self.snake[1:]:
            self._set_cell(body_segment[0], body_segment[1], EnvID.SNAKE_BODY)

    def _update_snake_body(self) -> None:
        for i in range(self.snake.shape[0] - 1):
            idx: int = -(i + 1)
            self.snake[idx] = self.snake[idx - 1]

    def _check_collision(self, id: EnvID) -> bool:
        if self.get_cell(self.snake[0][0], self.snake[0][1]) == id.value:
            return True
        return False

    def reset(self, w: int, h: int) -> None:
        self._env = np.zeros((h, w), dtype=int)
        self._env[0] = EnvID.WALL.value
        self._env[h - 1] = EnvID.WALL.value
        self._env[:, 0] = EnvID.WALL.value
        self._env[:, w - 1] = EnvID.WALL.value
        for _ in range(self._N_RED):
            self._spawn_obj(EnvID.APPLE_RED)
        for _ in range(self._N_GREEN):
            self._spawn_obj(EnvID.APPLE_GREEN)
        self._init_snake()
        for _ in range(self._SNAKE_LEN):
            self._extend_snake()
        self._snake_to_env()

    def step(self, action: Action) -> int:
        reward: QReward = QReward.OK
        self._update_snake_body()
        if action == action.UP:
            self.snake[0] += np.array([0, -1])
        elif action == action.DOWN:
            self.snake[0] += np.array([0, 1])
        elif action == action.LEFT:
            self.snake[0] += np.array([-1, 0])
        elif action == action.RIGHT:
            self.snake[0] += np.array([1, 0])
        if not self._check_collision(EnvID.EMPTY):
            if self._check_collision(EnvID.APPLE_RED):
                self._set_cell(self.snake[0][0], self.snake[0][1], EnvID.EMPTY)
                self._spawn_obj(EnvID.APPLE_RED)
                self.snake = self.snake[:-1]
                if self.snake.shape[0] == 0:
                    reward = QReward.DEAD
                else:
                    reward = QReward.LOSE
            elif self._check_collision(EnvID.APPLE_GREEN):
                self._set_cell(self.snake[0][0], self.snake[0][1], EnvID.EMPTY)
                self._spawn_obj(EnvID.APPLE_GREEN)
                self._extend_snake()
                reward = QReward.GAIN
            elif self._check_collision(EnvID.WALL) or self._check_collision(
                EnvID.SNAKE_BODY
            ):
                reward = QReward.DEAD
        self._snake_to_env()
        return reward

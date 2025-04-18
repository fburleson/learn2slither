from enum import Enum


class EnvID(Enum):
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


class QReward(Enum):
    OK = 0
    DEAD = 1
    GAIN = 2
    LOSE = 3


class RunModes(Enum):
    TRAIN = 0
    PLAY = 1

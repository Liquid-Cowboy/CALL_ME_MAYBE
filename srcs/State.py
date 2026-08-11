from enum import Enum, auto


class State(Enum):
    EXPECTS_START = auto(),
    EXPECTS_END = auto(),
    EXPECTS_KEY = auto(),
    EXPECTS_VALUE = auto(),
    EXPECTS_COLON = auto(),
    EXPECTS_COMMA_OR_END = auto(),

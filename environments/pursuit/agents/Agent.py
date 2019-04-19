import pathlib
from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, name, idx, evaluate):
        self.id = idx
        self.name = name
        self.exploit = evaluate
        self.train = not self.exploit

    @abstractmethod
    def act(self, state):
        raise NotImplementedError()

    @abstractmethod
    def feedback(self, state, actions, reward, next_state, terminal):
        raise NotImplementedError()

    def save(self, directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load(self, directory):
        raise NotImplementedError()

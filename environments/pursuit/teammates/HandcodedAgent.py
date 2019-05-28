from abc import ABC, abstractmethod


class HandcodedAgent(ABC):

    def __init__(self, name, idx):
        self.id = idx
        self.name = name

    @abstractmethod
    def act(self, state):
        raise NotImplementedError()

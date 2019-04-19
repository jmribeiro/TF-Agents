import random

from environments.pursuit.agents.Agent import Agent
from environments.pursuit.utils import action_pool


class DummyAgent(Agent):

    def save(self, directory):
        pass

    def load(self, directory):
        pass

    def __init__(self, idx):
        super().__init__("dummy", idx, True)

    def act(self, state):
        return action_pool()[random.randint(0, 3)]

    def feedback(self, state, actions, reward, next_state, terminal):
        pass

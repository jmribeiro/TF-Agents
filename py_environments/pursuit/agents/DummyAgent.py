import random

from py_environments.pursuit.agents.HandcodedAgent import HandcodedAgent
from py_environments.pursuit.utils import action_pool


class DummyAgent(HandcodedAgent):

    def __init__(self, idx):
        super().__init__("dummy", idx)

    def act(self, state):
        return action_pool()[random.randint(0, 3)]

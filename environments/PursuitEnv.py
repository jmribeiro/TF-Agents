import random

import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step as ts

from environments.pursuit.PursuitState import PursuitState
from environments.pursuit.PyGameVisualizer import PygameVisualizer
from environments.pursuit.teammates.DummyAgent import DummyAgent
from environments.pursuit.teammates.GreedyAgent import GreedyAgent
from environments.pursuit.teammates.ProbabilisticDestinations import ProbabilisticDestinationsAgent
from environments.pursuit.teammates.TeammateAwareAgent import TeammateAwareAgent
from environments.pursuit.utils import action_pool, move


class PursuitEnv(PyEnvironment):

    def __init__(self, team="greedy", world_size=(5, 5)):

        super().__init__()

        self.teammates = self._spawn_team(team)
        self.total_agents = len(self.teammates) + 1
        self.world_size = world_size

        self._action_spec = BoundedArraySpec(shape=(),
                                             dtype=np.int32,
                                             minimum=0, maximum=3,
                                             name='action')
        self._observation_spec = BoundedArraySpec(shape=(2 * self.total_agents,),
                                                  dtype=np.int32,
                                                  name='observation')
        self.visualizers = ()
        self.initialized_visualizers = self.visualizers != ()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action):

        assert 0 <= action <= 3, "`action` should be 0 (Left), 1 (Right), 2 (Up) or 3 (Down)"

        if self._terminal:
            for visualizer in self.visualizers: visualizer.end()
            return self.reset()

        if self._initial: self._initial = False

        # Build Joint Actions
        pursuit_action = action_pool()[action]
        joint_actions = [pursuit_action, ]
        for teammate in self.teammates: joint_actions.append(teammate.act(self._state))

        # Step
        next_state = self._transition_function(self._state, joint_actions)
        self._reward = PursuitEnv._reward_function(next_state)
        self._terminal = next_state.terminal
        self._state = next_state

        observation = PursuitEnv._distances_to_prey(self._state)
        if self._terminal:
            return ts.termination(observation, self._reward)
        else:
            return ts.transition(observation, self._reward, discount=1.0)

    def _reset(self):
        self._state = PursuitState.random_state(self.total_agents, self.world_size, random.Random(100))
        self._initial = True
        self._terminal = False
        observation = PursuitEnv._distances_to_prey(self._state)
        return ts.restart(observation)

    def render(self, mode='rgb_array'):

        if not self.initialized_visualizers:
            self._initialize_visualizers()

        if self._initial:
            for visualizer in self.visualizers: visualizer.start(self._state)

        for visualizer in self.visualizers: visualizer.update(self._state)

    def close(self):
        for visualizer in self.visualizers: visualizer.end()
        super().close()

    def _transition_function(self, state, joint_actions):

        assert (len(joint_actions) == self.total_agents)

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

        occupied_positions = set(state.prey_positions) | set(state.agent_positions)

        num_preys = len(state.prey_positions)

        apos_array = [None] * self.total_agents
        ppos_array = [None] * num_preys
        agents_indexs = [(i, True) for i in range(self.total_agents)] + \
                        [(i, False) for i in range(num_preys)]
        random.shuffle(agents_indexs)

        for i, is_agent in agents_indexs:
            if is_agent:
                position = state.agent_positions[i]
                action = joint_actions[i]
            else:
                position = state.prey_positions[i]
                action = random.choice(directions)
            new_position = move(position, action, self.world_size)

            # if collision is detected, just go to the original position
            if new_position in occupied_positions:
                new_position = position

            occupied_positions.remove(position)
            occupied_positions.add(new_position)

            if is_agent:
                apos_array[i] = new_position
            else:
                ppos_array[i] = new_position

        return PursuitState(prey_positions=tuple(ppos_array),
                            agent_positions=tuple(apos_array),
                            world_size=tuple(self.world_size))

    def _initialize_visualizers(self):
        assert not self.initialized_visualizers, "Visualizers already initialized"
        self.visualizers = (
            PygameVisualizer(400, 400,
                             agent_colors=[(200, 100, 255, 255)] + [(0, 0, 255, 255)
                                                                    for _ in range(self.total_agents - 1)]),
        )
        self.initialized_visualizers = True

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def _spawn_teammate(agent_type, idx):
        if agent_type == "dummy":
            agent = DummyAgent(idx)
        elif agent_type == "greedy":
            agent = GreedyAgent(idx)
        elif agent_type == "teammate aware":
            agent = TeammateAwareAgent(idx)
        elif agent_type == "probabilistic destinations":
            agent = ProbabilisticDestinationsAgent(idx)
        else:
            raise ValueError(f"Invalid Config: Unknown agent type {agent_type}")
        return agent

    @staticmethod
    def _spawn_team(team):
        if team == "mixed":
            teammates = ["teammate aware", "greedy", "probabilistic destinations"]
        elif team == "greedy":
            teammates = ["greedy", "greedy", "greedy"]
        elif team == "teammate aware":
            teammates = ["teammate aware", "teammate aware", "teammate aware"]
        else:
            raise ValueError(f"Unknown team {team}\nAvailable Teams are 'mixed', 'greedy' and 'teammate aware'")
        return [PursuitEnv._spawn_teammate(teammate, idx + 1) for idx, teammate in enumerate(teammates)]

    @staticmethod
    def _reward_function(state):
        return 100 if state.terminal else -1

    @staticmethod
    def _coordinates(state):
        return state.features().astype("int32")

    @staticmethod
    def _distances_to_prey(state):
        return state.features_relative_prey().astype("int32")

    @staticmethod
    def _distances_to_agent(state, agent_idx):
        return state.features_relative_agent(agent_idx).astype("int32")

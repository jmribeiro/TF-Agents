import random

import numpy as np

from environments.pursuit.utils import cornered, distance, directionx, directiony


class PursuitState(object):

    def __init__(self, agent_positions, prey_positions, world_size):
        assert (isinstance(agent_positions, tuple))
        assert (isinstance(prey_positions, tuple))
        assert (isinstance(world_size, tuple))
        assert (len(world_size) == 2)
        self.agent_positions = agent_positions
        self.prey_positions = prey_positions
        self.terminal = True
        self.world_size = world_size
        self.occupied = None
        for prey in prey_positions:
            if not cornered(self, prey, world_size):
                self.terminal = False
                break

    @property
    def occupied_cells(self):
        if not self.occupied:
            self.occupied = set(self.agent_positions) | set(self.prey_positions)
        return self.occupied

    @staticmethod
    def random_state(num_agents, world_size, random_instance=None):
        if random_instance is None:
            random_instance = random._inst

        assert (num_agents >= 4)
        world_size = tuple(world_size)
        num_preys = num_agents // 4

        assert (world_size[0] * world_size[1] > num_agents + num_preys)
        filled_positions = set()

        ppos_array = [(0, 0)] * num_preys
        apos_array = [(0, 0)] * num_agents
        for i in range(num_preys):
            while True:
                pos = (random_instance.randint(0, world_size[0] - 1), random_instance.randint(0, world_size[1] - 1))
                if pos not in filled_positions:
                    break

            ppos_array[i] = pos
            filled_positions.add(pos)

        for i in range(num_agents):
            while True:
                pos = (random_instance.randint(0, world_size[0] - 1), random_instance.randint(0, world_size[1] - 1))
                if pos not in filled_positions:
                    break

            apos_array[i] = pos
            filled_positions.add(pos)

        return PursuitState(prey_positions=tuple(ppos_array), agent_positions=tuple(apos_array), world_size=world_size)

    @staticmethod
    def from_features(features, world_size):
        features = features.reshape(-1, 2)
        agent_positions = tuple(tuple(pos) for pos in features[:4])
        prey_position = (tuple(features[4]),)
        return PursuitState(agent_positions=agent_positions, prey_positions=prey_position, world_size=world_size)

    @staticmethod
    def from_features_relative_prey(features, world_size):
        agent_positions = features.reshape(-1, 2)
        mid = (world_size[0] // 2, world_size[1] // 2)
        for i, (x, y) in enumerate(agent_positions):
            agent_positions[i] = (mid[0] + x, mid[1] + y)
        agent_positions = tuple(tuple(pos) for pos in agent_positions)
        return PursuitState(agent_positions=agent_positions, prey_positions=(mid,), world_size=world_size)

    def features(self):
        return np.concatenate((np.array(self.agent_positions), np.array(self.prey_positions))).reshape(-1)

    def features_relative_prey(self):
        """
        Tells each agent's distance to the prey
        """
        prey = self.prey_positions[0]
        relative_pos = []
        w, h = self.world_size
        for pos in self.agent_positions:
            relative_pos.append(self.distance_to(prey, pos, w, h))
        return np.concatenate(relative_pos)

    def features_relative_agent(self, agent_id):

        feature_array = []
        agent = self.agent_positions[agent_id]
        w, h = self.world_size

        # Teammates
        for i, teammate in enumerate(self.agent_positions):
            if i == agent_id: continue
            d = self.distance_to(agent, teammate, w, h)
            feature_array.append(d)

        # Sort (from closest to furthest)
        feature_array.sort(key=lambda dists: sum(abs(dists)))

        # Prey
        prey = self.prey_positions[0]
        d = self.distance_to(agent, prey, w, h)
        feature_array.append(d)

        return np.concatenate(feature_array)

    @staticmethod
    def distance_to(pivot, other, w, h):
        dx, dy = distance(pivot, other, w, h)
        dx = dx * directionx(pivot, other, w)
        dy = dy * directiony(pivot, other, h)
        return np.array([dx, dy])

    def __repr__(self):
        s = "Agents: " + ', '.join(str(p) for p in self.agent_positions)
        s += "\n"
        s += "Prey:" + ', '.join(str(p) for p in self.prey_positions)
        return s

    def __hash__(self):
        return hash(self.agent_positions + self.prey_positions)

    def __eq__(self, other):
        return self.agent_positions == other.agent_positions and \
               self.prey_positions == other.prey_positions and \
               self.world_size == other.world_size

    def __add__(self, offsets):
        assert (isinstance(offsets, np.ndarray))
        features = self.features()
        F = len(features)
        rows = self.world_size[0]
        columns = self.world_size[1]
        return PursuitState.from_features((features + offsets) % ([rows, columns] * (F // 2)), self.world_size)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        assert (isinstance(other, PursuitState))
        features = self.features()
        max_list = [self.world_size[0], self.world_size[1]] * (len(features) // 2)
        half_list = [value // 2 for value in max_list]
        return ((features - other.features()) + half_list) % max_list - half_list
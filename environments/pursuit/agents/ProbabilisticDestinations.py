import numpy as np

from environments.pursuit.agents.Agent import Agent
from environments.pursuit.utils import direction, move, action_pool, total_actions, manhattan_distance, softmax


class ProbabilisticDestinationsAgent(Agent):

    def __init__(self, idx):
        super().__init__("probabilistic destinations", idx, True)

    def feedback(self, state, actions, reward, next_state, terminal):
        pass

    def save(self, directory):
        pass

    def load(self, directory):
        pass

    def act(self, state):
        actions = action_pool()
        action_probs = np.zeros(total_actions())
        w, h = state.world_size
        my_pos = state.agent_positions[self.id]
        prey_pos = state.prey_positions[0]

        # don't go further than half the world
        max_dist = min(min(w, h) // 2, manhattan_distance(my_pos, prey_pos, w, h))
        # if im next to the prey, move onto it
        if max_dist == 1:
            return direction(my_pos, prey_pos, w, h)

        distances = np.arange(1, max_dist)
        distance_probs = softmax(distances, -1)
        for i in range(len(distances)):
            dist = distances[i]
            # all destinations at distance = dist from the prey, which are unblocked and which action should the
            # agent take
            dests, dest_actions = self.compute_destinations(dist, state)
            if len(dests) == 0:
                continue

            # distances between each destination and me
            dist_to_me = np.array([manhattan_distance(my_pos, dest, w, h) for dest in dests])

            dist_to_me_probs = softmax(dist_to_me, -1)
            for j, a in enumerate(dest_actions):
                action_probs[actions.index(a)] += dist_to_me_probs[j] * distance_probs[i]

        # if nothing available, move randomly to an unblocked cell
        if sum(action_probs) == 0:
            for a in action_pool():
                if move(my_pos, a, (w, h)) not in state.occupied_cells:
                    action_probs[action_pool().index(a)] = 1.0

        action_probs /= sum(action_probs)

        return actions[np.random.choice(np.arange(4), p=action_probs)]

    def compute_destinations(self, distance, state):
        w, h = state.world_size
        px, py = state.prey_positions[0]
        my_pos = state.agent_positions[self.id]
        all_dests = []
        all_actions = []

        def destinations():
            # from top to right
            for i in range(distance):
                yield ((px + i) % w, (py - distance + i) % h)

            # from right to bottom
            for i in range(distance):
                yield ((px + distance - i) % w, (py + i) % h)

            # from bottom to left
            for i in range(distance):
                yield ((px - i) % w, (py + distance - i) % h)

            # from left to top
            for i in range(distance):
                yield ((px - distance + i) % w, (py - i) % h)

        for dest in destinations():
            action = direction(my_pos, dest, w, h)
            if dest not in state.occupied_cells and move(my_pos, action, (w, h)) not in state.occupied_cells:
                all_dests.append(dest)
                all_actions.append(action)

        return all_dests, all_actions

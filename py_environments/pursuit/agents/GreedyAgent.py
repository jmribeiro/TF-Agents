import numpy as np

from py_environments.pursuit.agents.HandcodedAgent import HandcodedAgent
from py_environments.pursuit.utils import distance, direction, directionx, directiony, move, cornered, action_pool, \
    total_actions


class GreedyAgent(HandcodedAgent):

    def __init__(self, idx):
        super().__init__("greedy", idx)

    def act(self, state):
        A = total_actions()
        policy = self.policy(state)
        choice = np.random.choice(range(A), p=policy)
        return action_pool()[choice]

    def policy(self, state):

        actions = action_pool()
        A = total_actions()
        probs = np.zeros((A,))

        w, h = state.world_size
        my_pos = state.agent_positions[self.id]
        closest_prey, d = None, None
        for prey in state.prey_positions:
            distance_to_prey = sum(distance(my_pos, prey, w, h))
            # already neighboring some prey
            if distance_to_prey == 1:
                chosen_action = direction(my_pos, prey, w, h)
                probs[actions.index(chosen_action)] = 1.0
                return probs
            # get the closest non cornered prey
            if d is None or (not cornered(state, prey, (w, h)) and distance_to_prey < d):
                closest_prey, d = prey, distance_to_prey

        # unoccupied neighboring cells, sorted by proximity to agent
        targets = [move(closest_prey, d, (w, h)) for d in actions]
        targets = list(filter(lambda x: x not in state.occupied_cells, targets))

        if len(targets) == 0:
            for i in range(A):
                probs[i] = 1 / A
            return probs

        target = min(targets, key=lambda pos: sum(distance(my_pos, pos, w, h)))

        dx, dy = distance(my_pos, target, w, h)
        move_x = (directionx(my_pos, target, w), 0)
        move_y = (0, directiony(my_pos, target, h))
        pos_x = move(my_pos, move_x, (w, h))
        pos_y = move(my_pos, move_y, (w, h))

        # moving horizontally since there's a free cell
        if pos_x not in state.occupied_cells and (dx > dy or dx <= dy and pos_y in state.occupied_cells):
            action = move_x
            probs[actions.index(action)] = 1.0
            return probs
        # moving vertically since there's a free cell
        elif pos_y not in state.occupied_cells and (dx <= dy or dx > dy and pos_x in state.occupied_cells):
            action = move_y
            probs[actions.index(action)] = 1.0
            return probs
        # moving randomly since there are no free cells towards prey
        else:
            for i in range(A):
                probs[i] = 1 / A
            return probs

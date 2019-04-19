import numpy as np

from environments.pursuit.agents.Agent import Agent
from environments.pursuit.agents.decisionmaking.astar import astar
from environments.pursuit.utils \
    import distance, direction, move, cornered, action_pool, total_actions, argmin, argmax


class TeammateAwareAgent(Agent):

    def feedback(self, state, actions, reward, next_state, terminal):
        pass

    def save(self, directory):
        pass

    def load(self, directory):
        pass

    def __init__(self, id):
        super().__init__("teammate aware", id, True)
        self.last_prey_pos = None
        self.prey_id = None
        self.last_target = None

    def act(self, state):
        A = total_actions()
        policy = self.policy(state)
        choice = np.random.choice(range(A), p=policy)
        return action_pool()[choice]

    def policy(self, state):

        actions = action_pool()
        A = total_actions()

        my_pos = state.agent_positions[self.id]
        w, h = state.world_size

        def choose_action():

            probs = np.zeros((A,))

            target = self.last_target
            # if already at destination, just follow the prey
            if my_pos == target:
                action = direction(my_pos, self.last_prey_pos, w, h)
                probs[actions.index(action)] = 1.0
                return probs

            action, dist = astar(my_pos, state.occupied_cells - {target}, target, (w, h))

            if action is None:
                for i in range(A):
                    probs[i] = 1 / A
                return probs
            else:
                probs[actions.index(action)] = 1.0
                return probs

        closest_prey, d, prey_id = None, None, 0
        for i, prey in enumerate(state.prey_positions):
            distance_to_prey = sum(distance(my_pos, prey, w, h))
            # get the closest non cornered prey
            if d is None or (not cornered(state, prey, (w, h)) and distance_to_prey < d):
                closest_prey, d, prey_id = prey, distance_to_prey, i

        self.prey_id = prey_id
        self.last_prey_pos = state.prey_positions[self.prey_id]
        # get the 4 agents closest to the prey
        # agents = sorted(state.agent_positions, key=lambda p: sum(distance(p, closest_prey, w, h)))
        # agents = agents[:4]
        agents = state.agent_positions

        # sort the agents by the worst shortest distance to the prey
        neighboring = [move(closest_prey, d, (w, h)) for d in actions]
        distances = [[sum(distance(a, p, w, h)) for p in neighboring] for a in agents]
        # distances = [(sorted((astar_distance(p, n, state.occupied_cells, (w, h)), i) for i, n in enumerate(neighboring)), j) for j, p in enumerate(agents)]

        # distances[i][j] is the distance of agent i to cell j
        # taken = set()
        target = 0
        for _ in range(len(agents)):
            min_dists = [min(d) for d in distances]
            min_inds = [argmin(d) for d in distances]
            selected_agent = argmax(min_dists)
            target = min_inds[selected_agent]
            # print('%d selected for %d' % (selected_agent, target))
            if selected_agent == self.id:
                break
            # remove the target from other agents
            for d in distances:
                d[target] = 2 ** 31
            # remove the agent itself
            for i in range(len(distances[selected_agent])):
                distances[selected_agent][i] = -1

        self.last_target = neighboring[target]

        return choose_action()

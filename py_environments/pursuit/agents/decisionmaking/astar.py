from heapq import *

from py_environments.pursuit.utils import neighbors, distance, direction


class Node(object):
    def __init__(self, position, parent, cost, heuristic):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic

    def __hash__(self):
        return self.position.__hash__()

    def __eq__(self, other):
        return self.position == other.position


def astar(initial_pos, obstacles, target, world_size):
    if initial_pos == target:
        return (0, 0), 0
    w, h = world_size
    obstacles = obstacles - {target}

    def heuristic(pos):
        return sum(distance(initial_pos, pos, w, h))

    # each item in the queue contains (heuristic+cost, cost, position, parent)
    initial_node = Node(initial_pos, None, 0, heuristic(initial_pos))
    queue = [Node(n, initial_node, 1, sum(distance(n, target, w, h)))
             for n in neighbors(initial_pos, world_size) if n not in obstacles]

    heapify(queue)
    visited = set()
    visited.add(initial_pos)
    current = initial_node

    while len(queue) > 0:
        current = heappop(queue)

        if current.position in visited:
            continue

        visited.add(current.position)

        if current.position == target:
            break

        for position in neighbors(current.position, world_size):
            if position not in obstacles:
                new_node = Node(position, current, current.cost + 1, heuristic(position))
                heappush(queue, new_node)

    if target not in visited:
        return None, w * h

    i = 1
    while current.parent != initial_node:
        current = current.parent
        i += 1

    return direction(initial_pos, current.position, h, w), i


def astar_distance(source, target, occupied_cells, world_size):
    _, d = astar(source, occupied_cells, target, world_size)
    return d

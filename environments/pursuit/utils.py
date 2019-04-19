def action_pool():
    return [(1, 0), (-1, 0), (0, 1), (0, -1)]


def total_actions():
    return 4


def neighbors(pos, world_size):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    result = []
    for d in directions:
        result.append(move(pos, d, world_size))
    return result


def move(pos, d, world_size):
    return (pos[0] + d[0]) % world_size[0], (pos[1] + d[1]) % world_size[1]


def distance(pos1, pos2, w, h):
    dx = min((pos1[0] - pos2[0]) % w, (pos2[0] - pos1[0]) % w)
    dy = min((pos1[1] - pos2[1]) % h, (pos2[1] - pos1[1]) % h)
    return dx, dy


def manhattan_distance(pos1, pos2, w, h):
    return sum(distance(pos1, pos2, w, h))


def softmax(array, factor=1.0):
    import numpy as np
    array = array * factor
    array = np.exp(array - np.max(array))
    return array / array.sum()


def direction(source, target, w, h):
    dx_forward = (target[0] - source[0]) % w
    dx_backward = (source[0] - target[0]) % w
    dy_forward = (target[1] - source[1]) % h
    dy_backward = (source[1] - target[1]) % h

    if dx_forward < dx_backward:
        return 1, 0
    elif dx_backward < dx_forward:
        return -1, 0
    elif dy_forward < dy_backward:
        return 0, 1
    elif dy_backward < dy_forward:
        return 0, -1
    else:
        return 0, 0


def directionx(source, target, w):
    dx_forward = (target[0] - source[0]) % w
    dx_backward = (source[0] - target[0]) % w

    if dx_forward < dx_backward:
        return 1
    elif dx_backward < dx_forward:
        return -1
    else:
        return 0


def directiony(source, target, h):
    dy_forward = (target[1] - source[1]) % h
    dy_backward = (source[1] - target[1]) % h

    if dy_forward < dy_backward:
        return 1
    elif dy_backward < dy_forward:
        return -1
    else:
        return 0


def cornered(state, position, world_size):
    for n in neighbors(position, world_size):
        if n not in state.occupied_cells:
            return False
    return True


def argmin(arr):
    if len(arr) == 0:
        return None
    result = 0
    for i in range(len(arr)):
        if arr[i] < arr[result]:
            result = i
    return result


def argmax(arr):
    if len(arr) == 0:
        return None
    result = 0
    for i in range(len(arr)):
        if arr[i] > arr[result]:
            result = i
    return result

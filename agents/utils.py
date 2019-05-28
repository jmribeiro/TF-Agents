from collections import namedtuple

Datapoint = namedtuple("Datapoint", "state action reward next_state terminal info")

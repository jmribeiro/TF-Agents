from tf_agents.drivers.driver import Driver


class MidTrainingEvaluatorDriver(Driver):

    def __init__(self, env, policy, observers=None, num_steps=1,):
        super(MidTrainingEvaluatorDriver, self).__init__(env, policy)

    def run(self):
        # TODO
        pass

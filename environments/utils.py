import tensorflow as tf
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.environments import utils
from tf_agents.metrics.py_metrics import AverageReturnMetric
from tf_agents.policies.random_py_policy import RandomPyPolicy

tf.compat.v1.enable_v2_behavior()


def validate_environment(env):

    # 1 - Validate Using Utils
    utils.validate_py_environment(env, episodes=5)

    # 2 - Test with Driver & Random Policy
    random_policy = RandomPyPolicy(time_step_spec=env.time_step_spec(), action_spec=env.action_spec())
    metric = AverageReturnMetric()
    driver = PyDriver(env, random_policy, [metric, ], max_steps=100, max_episodes=5)
    time_step_0 = env.reset()
    time_step_T, _ = driver.run(time_step_0)
    print(f"Successfully Validated Environment: Average Return: {metric.result()}")
    env.close()

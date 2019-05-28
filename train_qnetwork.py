import yaml
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.metrics.tf_metrics import NumberOfEpisodes, AverageReturnMetric
from tf_agents.metrics.tf_metrics import EnvironmentSteps
import tensorflow as tf
from agents.ReplayDQNAgent import ReplayDQNAgent
from environments.CardGameEnv import CardGameEnv

if __name__ == '__main__':

    # Configuration Reading
    with open("config.yaml", "r") as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    timesteps = config["timesteps"]
    env_name = config["environment"]
    evaluation_interval = config["evaluation interval"]
    evaluation_episodes = config["evaluation episodes"]
    layers = [int(layer) for layer in config["layers"]]
    learning_rate = config["learning rate"]
    replay_buffer_size = config["replay buffer size"]
    replay_batch_size = config["batch size"]
    render = config["render"]

    tf.compat.v1.enable_resource_variables()
    tf.compat.v1.enable_eager_execution()

    # Environment
    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)
    train_env = TFPyEnvironment(train_py_env)
    eval_env = TFPyEnvironment(eval_py_env)

    # Agent
    agent = ReplayDQNAgent(train_env.time_step_spec(), train_env.action_spec(), train_env.action_spec(),
                           replay_buffer_size, replay_batch_size, train_env.batch_size)

    avg_return = AverageReturnMetric()
    num_episodes = NumberOfEpisodes()
    env_steps = EnvironmentSteps()

    metrics = [avg_return, num_episodes, env_steps]

    driver = DynamicStepDriver(train_env, agent.collect_policy, metrics, timesteps)

    while True:
        final_timestep, policy_state = driver.run()
        for metric in metrics:
            print(f"{metric.name}: {metric.result().numpy()}")

import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.trajectories.trajectory import from_transition
from tf_agents.utils import common

from agents.ReplayDqnAgent import ReplayDqnAgent
from py_environments.CardGameEnv import CardGameEnv
from py_environments.PursuitEnv import PursuitEnv

tf.compat.v1.enable_v2_behavior()


def setup_environments(env_name):
    train_env = TFPyEnvironment(suite_gym.load(env_name))
    eval_env = TFPyEnvironment(suite_gym.load(env_name))
    return train_env, eval_env


def interact(agent, env, render):

    pi = agent.collect_policy

    time_step = env.current_time_step()
    action_step = pi.action(time_step)
    next_time_step = env.step(action_step.action)

    datapoint = from_transition(time_step, action_step, next_time_step)
    agent.feedback(datapoint)

    if render:
        env.pyenv.envs[0].render()


def evaluate_policy(env, pi, num_episodes):

    total_return = 0.0

    for _ in range(num_episodes):

        time_step = env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            policy = pi.action(time_step)
            time_step = env.step(policy.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def train(timesteps, agent, replay_batch_size, train_env, eval_env, eval_interval, eval_episodes, render):
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = evaluate_policy(eval_env, agent.policy, eval_episodes)
    print('step = {0}: Average Return = {1}'.format(0, avg_return))
    returns = [avg_return]

    for t in range(timesteps):

        # Collect one step using collect_policy and save to the replay buffer.
        interact(agent, train_env, render)

        if t > replay_batch_size:

            experience, unused_info = next(agent.replay_memory_iterator)
            agent.train(experience)
            step = agent.train_step_counter.numpy()

            if step % eval_interval == 0:
                avg_return = evaluate_policy(eval_env, agent.policy, eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

    return returns


def plot_run(timesteps, eval_interval, returns):
    steps = range(0, timesteps + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim(top=250)
    return plt


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

    # Environment
    train_env, eval_env = setup_environments(env_name)

    #TODO - Register Environments for suite loading
    #train_env, eval_env = TFPyEnvironment(PursuitEnv("teammate aware")), TFPyEnvironment(PursuitEnv("teammate aware"))
    #train_env, eval_env = TFPyEnvironment(CardGameEnv()), TFPyEnvironment(CardGameEnv())

    # Agent
    agent = ReplayDqnAgent(train_env.time_step_spec(), train_env.observation_spec(), train_env.action_spec(),
                           replay_buffer_size, replay_batch_size, train_env.batch_size,
                           layers, learning_rate)

    # Train and Evaluate
    returns = train(
        timesteps,
        agent,
        replay_batch_size,
        train_env,
        eval_env,
        evaluation_interval,
        evaluation_episodes,
        render
    )

    print(returns)

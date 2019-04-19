import yaml

import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.dqn import q_network
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()


def setup_environments(env_name):
    train_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    eval_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    return train_env, eval_env


def setup_agent(env, layers, learning_rate):

    time_step_spec = env.time_step_spec()
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    network = q_network.QNetwork(observation_spec, action_spec, fc_layer_params=layers)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_step_counter = tf.compat.v2.Variable(0)
    agent = dqn_agent.DqnAgent(time_step_spec, action_spec, network, optimizer,
                               train_step_counter=train_step_counter,
                               td_errors_loss_fn=dqn_agent.element_wise_squared_loss)
    agent.initialize()

    return agent


def interact(env, pi, replay_buffer, render):
    time_step = env.current_time_step()
    action_step = pi.action(time_step)
    next_time_step = env.step(action_step.action)
    datapoint = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(datapoint)
    if render:
        env.pyenv.envs[0].render()


def setup_replay_buffer(max_size, agent, batch_size):
    data_spec = agent.collect_data_spec
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, batch_size, max_size)
    dataset = replay_buffer.as_dataset(batch_size, num_steps=2, num_parallel_calls=3).prefetch(3)
    iterator = iter(dataset)
    return replay_buffer, iterator


def evaluate_policy(env, pi, num_episodes=10):

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


def train(timesteps, agent, replay_buffer, iterator, batch_size, train_env, eval_env, eval_interval, eval_episodes, render):

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
        interact(train_env, agent.collect_policy, replay_buffer, render)

        if t > batch_size:

            experience, unused_info = next(iterator)
            agent.train(experience)
            step = agent.train_step_counter.numpy()

            if step % eval_interval == 0:
                avg_return = evaluate_policy(eval_env, agent.policy, eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

    return returns


def plot_run(num_episodes, eval_interval, returns):
    steps = range(0, num_episodes + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim(top=250)
    return plt


def main():

    with open("config.yaml", "r") as stream:
        config = yaml.load(stream)

    train_env, eval_env = setup_environments(config["environment"])
    agent = setup_agent(train_env, [int(layer) for layer in config["layers"]], config["learning rate"])
    replay_buffer, iterator = setup_replay_buffer(config["replay buffer size"], agent, config["batch size"])

    returns = train(
        config["episodes"],
        agent,
        replay_buffer,
        iterator,
        config["batch size"],
        train_env,
        eval_env,
        config["evaluation interval"],
        config["evaluation episodes"],
        config["render"]
    )

    print(returns)


if __name__ == '__main__':
    main()

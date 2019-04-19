import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()


def setup_environments(env_name):
    train_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    eval_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    return train_env, eval_env


def setup_qnetwork_agent(time_step_spec, observation_spec, action_spec, layers, learning_rate):
    network = q_network.QNetwork(observation_spec, action_spec, fc_layer_params=layers)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_step_counter = tf.compat.v2.Variable(0)
    agent = dqn_agent.DqnAgent(time_step_spec, action_spec, network, optimizer,
                               train_step_counter=train_step_counter,
                               td_errors_loss_fn=dqn_agent.element_wise_squared_loss)
    agent.initialize()

    replay_buffer, iterator = setup_replay_buffer(replay_buffer_size,
                                                  replay_batch_size,
                                                  agent.collect_data_spec, train_env.batch_size)
    agent.replay_buffer = replay_buffer
    agent.iterator = iterator

    return agent


def setup_replay_buffer(max_size, replay_batch_size, data_spec, data_batch_size):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, data_batch_size, max_size)
    dataset = replay_buffer.as_dataset(replay_batch_size, num_steps=2, num_parallel_calls=3).prefetch(3)
    iterator = iter(dataset)
    return replay_buffer, iterator


def interact(env, pi, replay_buffer, render):
    time_step = env.current_time_step()
    action_step = pi.action(time_step)
    next_time_step = env.step(action_step.action)
    datapoint = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(datapoint)
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


def train(timesteps, agent, replay_buffer, iterator, batch_size, train_env, eval_env, eval_interval, eval_episodes,
          render):
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


if __name__ == '__main__':
    # Configuration Reading
    with open("config.yaml", "r") as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    episodes = config["episodes"]
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

    # Agent
    agent = setup_qnetwork_agent(train_env.time_step_spec(),
                                 train_env.observation_spec(),
                                 train_env.action_spec(),
                                 layers, learning_rate)

    # Train and Evaluate
    returns = train(
        episodes,
        agent,
        agent.replay_buffer,
        agent.iterator,
        replay_batch_size,
        train_env,
        eval_env,
        evaluation_interval,
        evaluation_episodes,
        render
    )

    print(returns)

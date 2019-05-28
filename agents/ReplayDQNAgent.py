import tensorflow as tf
from tf_agents.agents import DqnAgent
from tf_agents.agents.dqn.dqn_agent import element_wise_squared_loss
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer


class ReplayDQNAgent(DqnAgent):

    def __init__(self, time_step_spec, observation_spec, action_spec,
                 replay_memory_size, replay_memory_batch_size, env_batch_size,
                 q_network_layers=(64,), alpha=0.0001, gamma=1.0,
                 epsilon_greedy=0.1, boltzmann_temperature=None):

        q_network = QNetwork(observation_spec, action_spec, fc_layer_params=q_network_layers)
        optimizer = tf.compat.v1.train.AdamOptimizer(alpha)
        train_step_counter = tf.compat.v2.Variable(0)

        super().__init__(time_step_spec, action_spec, q_network, optimizer,
                         gamma=gamma,
                         epsilon_greedy=epsilon_greedy,
                         boltzmann_temperature=boltzmann_temperature,
                         train_step_counter=train_step_counter,
                         td_errors_loss_fn=element_wise_squared_loss)

        self.replay_memory_size = replay_memory_size
        self.replay_memory_batch_size = replay_memory_batch_size
        self.env_batch_size = env_batch_size

        self._initialize()

    def _initialize(self):
        super()._initialize()
        self._initialize_replay_memory()

    def _initialize_replay_memory(self):
        self.replay_buffer = TFUniformReplayBuffer(self.collect_data_spec, self.env_batch_size, self.replay_memory_size)
        dataset = self.replay_buffer.as_dataset(self.replay_memory_batch_size,
                                                num_steps=2, num_parallel_calls=3).prefetch(3)
        self.replay_memory_iterator = iter(dataset)

    def reinforce(self, datapoint):
        replay_buffer = self.replay_buffer
        replay_buffer.add_batch(datapoint)

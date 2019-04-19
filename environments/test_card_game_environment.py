import tensorflow as tf
from tf_agents.environments import utils
from environments.CardGameEnv import CardGameEnv
tf.compat.v1.enable_v2_behavior()

"""
    CardGameEnv testing from https://github.com/tensorflow/agents colab tutorials
"""

environment = CardGameEnv()
utils.validate_py_environment(environment, episodes=5)

get_new_card_action = 0
end_round_action = 1

time_step = environment.reset()
print(time_step)
cumulative_reward = time_step.reward

for _ in range(3):
    time_step = environment.step(get_new_card_action)
    print(time_step.observation)
    print(time_step.reward)
    cumulative_reward += time_step.reward

time_step = environment.step(end_round_action)
cumulative_reward += time_step.reward
print('Final Reward = ', cumulative_reward)

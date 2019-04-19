import random
import time
from tf_agents.environments import utils
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from environments.PursuitEnv import PursuitEnv
import tensorflow as tf

tf.compat.v1.enable_v2_behavior()

# 1 - Validate Using Utils
environment = PursuitEnv()
utils.validate_py_environment(environment, episodes=5)

# 2 - Test with Rendering and Random Policy

time_step = environment.reset()
environment.render()

random_policy = RandomTFPolicy(environment.time_step_spec(), environment.action_spec())

cumulative_reward = time_step.reward

while not time_step.is_last():
    # FIXME - Getting Tensorflow error when calling random_policy.action(time_step)
    # For now using random.choice to simulate RandomTFPolicy
    # action_step = random_policy.action(time_step)
    # action = action_step.action
    time_step = environment.step(random.choice(range(4)))
    environment.render()
    print(time_step.observation)
    cumulative_reward += time_step.reward

print('Cumulative Reward = ', int(cumulative_reward))
print('Steps to Capture = ', int(100 - cumulative_reward))

# Sleep 3 seconds to verify caught prey
print("Waiting 3 seconds before exiting")
time.sleep(1.0)
print("\r> 3", end='')
time.sleep(1.0)
print("\r> 2", end='')
time.sleep(1.0)
print("\r> 1", end='')
time.sleep(1.0)
print("\rGoodbye!")
environment.close()

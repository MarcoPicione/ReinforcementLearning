from custom_environment import CustomEnvironment
from pettingzoo.test import parallel_api_test
import time

if __name__ == "__main__":
    env = CustomEnvironment(render_mode="human")
    parallel_api_test(env, num_cycles=1000000)
    env.unwrapped.render_mode = "human"
    observations, infos = env.reset()

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        # print(observations)
        env.render()
        time.sleep(0.5)
    env.close()
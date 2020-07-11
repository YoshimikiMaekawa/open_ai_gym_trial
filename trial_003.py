import gym
import numpy as np

env = gym.make('MountainCar-v0')

def main():
    q_table = np.zeros((40, 40, 3))
    rewards = []

    for episode in range(10000):
        if episode % 100 == 0:
            print("Episode: {0}".format(episode))
        observation = env.reset()
        total_reward = 0
        for _ in range(200):
            # env.render()
            # print(observation)
            action = get_action(env, q_table, observation, episode)
            next_observation, reward, done, _ = env.step(action)
            q_table = update_q_table(q_table, action, observation, next_observation, reward, episode)
            total_reward += reward
            observation = next_observation
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                rewards.append(total_reward)
                break
    
    observation = env.reset()
    for _ in range(200):
        env.render()
        action = get_action(env, q_table, observation, episode)
        next_observation, reward, done, _ = env.step(action)
        observation = next_observation
        if done:
            break

    env.close()

def get_status(_observation):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / 40
    position = int((_observation[0] - env_low[0]) / env_dx[0])
    velocity = int((_observation[1] - env_low[1]) / env_dx[1])
    return position, velocity

def update_q_table(_q_table, _action, _observation, _next_observation, _reward, _episode):
    alpha = 0.2
    gamma = 0.99

    next_position, next_velocitu = get_status(_next_observation)
    next_max_q_value = max(_q_table[next_position][next_velocitu])

    position, velocity = get_status(_observation)
    q_value = _q_table[position][velocity][_action]

    _q_table[position][velocity][_action] = q_value + alpha * (_reward + gamma * next_max_q_value - q_value)

    return _q_table

def get_action(_env, _q_table, _observation, _episode):
    epsilon = 0.002
    if np.random.uniform(0, 1) > epsilon:
        position, velocity = get_status(_observation)
        action = np.argmax(_q_table[position][velocity])
    else:
        action = np.random.choice([0, 1, 2])
    return action

if __name__ == '__main__':
    main()
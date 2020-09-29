import numpy as np
from constricted_env import *
import datetime

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10_000_000
SAVE_EVERY = 10_000
EPSILON = 1
EPSILON_DECAY = 0.9975
MIN_EPSILON = 0.002
DISCRETE_SIZE = np.array([20, 30, 7, 7])
HIGHS = np.array([400, 600, 140, 140])
LOWS = np.array([-400, 0, -140, -140])
win_size = (HIGHS - LOWS) / DISCRETE_SIZE
# q_table = np.random.uniform(-2, 0, size=(20, 30, 7, 7, 6))
q_table = np.load('./q_tables/qtable292055.npy')

time_created = datetime.datetime.now().timetuple()


def correct_state(state):
    x, y, xspeed, yspeed = state
    if x < -390:
        x = -390
    elif x > 390:
        x = 390

    if y > 520:
        y = 520
    elif y < 0:
        y = 0

    if xspeed < -140:
        xspeed = -140
    elif xspeed >= 140:
        xspeed = 139

    if yspeed < -140:
        yspeed = -140
    elif yspeed >= 140:
        yspeed = 139

    return x, y, xspeed, yspeed


def get_discrete_state(state):
    discrete_state = (state - LOWS) / win_size
    return tuple(discrete_state.astype(np.int))


def get_reward(state, done):
    x, y, xspeed, yspeed = state

    if done:
        if 9 <= x <= 10 and y <= 0:
            return 200
        else:
            return -100
    else:
        distance_reward = -100 * np.sqrt(x ** 2 + y ** 2)
        speed_reward = -100 * np.sqrt(xspeed ** 2 + yspeed ** 2)
        return distance_reward + speed_reward


def get_movement_variables(action):
    if action == 0:
        left = False
        boost = False
        right = False
    elif action == 1:
        left = True
        boost = False
        right = False
    elif action == 2:
        left = False
        boost = True
        right = False
    elif action == 3:
        left = False
        boost = False
        right = True
    elif action == 4:
        left = True
        boost = True
        right = False
    elif action == 5:
        left = False
        boost = True
        right = True
    return left, boost, right


if __name__ == '__main__':
    wins = 0
    env = LunarLander()
    # episode = 0
    # while True:
    #     episode +=1
    rewards = []
    for episode in range(EPISODES):

        if episode % SAVE_EVERY == 0 and episode != 0:
            print(f"Episodes played: {episode}")
            print(f"Wins earned: {wins}")
            print(f"Average reward: {np.mean(rewards)}")
            rewards = []
            np.save(f'./q_tables/qtable{time_created[2]}{time_created[3]}{time_created[4]}.npy', q_table)
            print(f'models saved with id: {time_created[2]}{time_created[3]}{time_created[4]}')
        env.reset()

        state, _, done = env.step((boost, left, right))
        state = correct_state(state)
        discrete_state = get_discrete_state(state)

        while not done:
            # if games_played > RENDER_AFTER:
            #     env.render()

            if np.random.random() > EPSILON:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
                # Get random action
                action = np.random.randint(0, 5)
            boost, left, right = get_movement_variables(action)

            new_state, _, done = env.step((boost, left, right))
            new_state = correct_state(state)
            new_discrete_state = get_discrete_state(new_state)

            reward = get_reward(new_discrete_state, done)
            rewards.append(reward)
            if reward == 200:
                wins += 1

            if not done:
                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(q_table[new_discrete_state])

                # Current Q value (for current state and performed action)
                current_q = q_table[discrete_state + (action,)]

                # And here's our equation for a new Q value for current state and action
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                # Update Q table with new Q value
                q_table[discrete_state + (action,)] = new_q

                # Simulation ended (for any reason) - if goal position is achived - update Q value with reward directly
            else:
                q_table[discrete_state + (action,)] = reward

            discrete_state = new_discrete_state

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)

    env.close()


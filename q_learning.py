import datetime

from easy_env import *

LEARNING_RATE = 0.1
DISCOUNT = 0.95
DISCRETE_SIZE = np.array([20, 30, 10, 10])
HIGHS = np.array([400, 600, 140, 140])
LOWS = np.array([-400, 0, -140, -140])
win_size = (HIGHS - LOWS) / DISCRETE_SIZE
# q_table = np.random.uniform(-2, 0, size=(20,30,10,10,6))
q_table = np.load('./q_tables/q_table1.npy')
EPISODES = 10

EPSILON = 1  # not a constant, qoing to be decayed
EPSILON_DECAY = 0.9975
MIN_EPSILON = 0.001

time_created = datetime.datetime.now().timetuple()


def get_discrete_state(state):
    discrete_state = (state - LOWS) / win_size
    return tuple(discrete_state.astype(np.int))


def get_reward(state):
    x, y, xspeed, yspeed = state

    if 9 <= x <= 10 and y <= 0:
        return 200
    else:
        return -0.3


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
    env = LunarLander()

    games_played = 0
    game_trained = 0
    steps = 0
    while True:
        games_played += 1
        if games_played % 10 == 0:
            np.save(f'./q_tables/table{time_created[2]}{time_created[3]}{time_created[4]}.npy', q_table)
            print(f'models saved with id: {time_created[2]}{time_created[3]}{time_created[4]}')
        env.reset()
        (x, y, xspeed, yspeed), _, _ = env.step((boost, left, right))
        state = (x, y, xspeed, yspeed)
        discrete_state = get_discrete_state(state)

        done = False
        while not done:
            steps += 1
            # if games_played > RENDER_AFTER:
            #     env.render()
            # env.render()

            if np.random.random() > EPSILON:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
                # Get random action
                action = np.random.randint(0, 5)
            boost, left, right = get_movement_variables(action)

            (x, y, xspeed, yspeed), _, _ = env.step((boost, left, right))
            new_state = (x, y, xspeed, yspeed)
            new_discrete_state = get_discrete_state(new_state)

            reward = get_reward(new_discrete_state)
            if reward == 200:
                done = True

            if not done:
                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(q_table[new_discrete_state])

                # Current Q value (for current state and performed action)
                current_q = q_table[discrete_state + (action,)]

                # And here's our equation for a new Q value for current state and action
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                # Update Q table with new Q value
                q_table[discrete_state + (action,)] = new_q

                # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
            else:
                q_table[discrete_state + (action,)] = reward

            discrete_state = new_discrete_state

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)

    env.close()


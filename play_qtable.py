import numpy
from easy_env import *
LOWS = np.array([-400, 0, -140, -140])
DISCRETE_SIZE = np.array([20, 30, 10, 10])
HIGHS = np.array([400, 600, 140, 140])
win_size = (HIGHS - LOWS)/DISCRETE_SIZE
def get_discrete_state(state):
    discrete_state = (state - LOWS)/win_size
    return tuple(discrete_state.astype(np.int))

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

q_table = np.load('q_tables/q_table1.npy')

env = LunarLander()
env.reset()
exit_program = False
while not exit_program:
    env.render()
    (x, y, xspeed, yspeed), reward, done = env.step((boost, left, right))
    state = (x, y, xspeed, yspeed)
    discrete_state = get_discrete_state(state)
    action = np.argmax(q_table[discrete_state])
    left, boost, right = get_movement_variables(action)
    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                exit_program = True
            if event.key == pygame.K_UP:
                boost = True
            if event.key == pygame.K_DOWN:
                boost = False
            if event.key == pygame.K_RIGHT:
                left = False if right else True
                right = False
            if event.key == pygame.K_LEFT:
                right = False if left else True
                left = False
            if event.key == pygame.K_r:
                boost = False
                left = False
                right = False
                env.reset()

env.close()
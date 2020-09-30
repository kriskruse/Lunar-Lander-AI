from constricted_env import *
DISCRETE_SIZE = np.array([20, 30, 7, 7])
HIGHS = np.array([400, 600, 140, 140])
LOWS = np.array([-400, 0, -140, -140])
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


q_table = np.load('q_tables/qtable292234.npy')

env = LunarLander()
env.reset()
exit_program = False
while not exit_program:
    env.render()
    (x, y, xspeed, yspeed), reward, done = env.step((boost, left, right))
    state = (x, y, xspeed, yspeed)
    state = correct_state(state)
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
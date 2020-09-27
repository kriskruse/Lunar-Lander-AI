from LunarLander import *
from agent_q import DQN, Network
import torch

model = Network()
model.load_state_dict(torch.load('./weights/agent271822.tar'))  #random test 27189, Hård parameter test u fuel 27184, hård para test med fuel 271822
# god rand agent271739

env = LunarLander()
env.reset()
exit_program = False
while not exit_program:
    env.render()
    (x, y, xspeed, yspeed), reward, done = env.step((boost, left, right))
    state = (x, y, xspeed, yspeed)
    model_input = DQN.get_model_input(state)
    q_values = model(model_input)
    action = torch.argmax(q_values)
    left, boost, right = DQN.get_movement_variables(action)
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
    # print(x == env.rocket.x)

    if done:
        env.reset()

env.close()

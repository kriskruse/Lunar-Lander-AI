# Lunar Lander: AI-controlled play

# Instructions:
#   Land the rocket on the platform within a distance of plus/minus 20, 
#   with a horizontal and vertical speed less than 20
#
# Controlling the rocket:
#    arrows  : Turn booster rockets on and off
#    r       : Restart game
#    q / ESC : Quit

from LunarLander import *
import time
import numpy

env = LunarLander()
env.reset()
exit_program = False
episodes = 100000
render = False
# while not exit_program:
won = 0
lost = 0
end_position = []
rewards = []
time_taken = []
for i in range(episodes):
    env.reset()
    done = False
    start_time = time.time()
    while not done:
        if render:
            env.render()
        (x, y, xspeed, yspeed), reward, done = env.step((boost, left, right))

        boost = False
        left = False
        right = False

        max_speed = min(40, max(abs(x), 10))
        x_limit = 15

        if xspeed > max_speed:
            right = True
        if xspeed < -max_speed:
            left = True

        if x > x_limit and not left:
            right = True
        elif x < - x_limit and not right:
            left = True


        # maksimal y fart
        if yspeed > 60:
            boost = True

        # boost grænse
        if y < 120:
            boost = True

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
    if abs(x) <= 20 and abs(xspeed) <= 20 and abs(yspeed) <= 20:
        won += 1
    else:
        lost += 1

    time_taken.append((time.time() - start_time)*1000)

    rewards.append(reward)
    end_position.append((x,y))
np.save('won.npy', won)
np.save('lost.npy', lost)
np.save('rewards.npy', rewards)
np.save('time_taken.npy', time_taken)
np.save('end_position.npy', end_position)

env.close()

# x er positiv til højre for platformen og negativ til venstre for
# xspeed er positiv gående mod højre og negativ gående mod venstre

# hvor lang tid tager det
# gennemsnitlig score
# hvor præcist den lander
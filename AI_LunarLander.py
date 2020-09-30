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

env = LunarLander()
env.reset()
exit_program = False

while not exit_program:

    env.render()
    (x, y, xspeed, yspeed), reward, done = env.step((boost, left, right))

    boost = False
    left = False
    right = False

    max_speed = 40
    x_limit = 20


    if xspeed > max_speed:
        right = True
    if xspeed < -max_speed:
        left = True

    if x > x_limit and not left and xspeed > -20 * max(abs(x) // 50, 1):
        right = True
    elif x < -x_limit and not right and xspeed < 20 * max(abs(x) // 50,1):
        left = True

    if -15 <= x <= 15:
        left = False
        right = False
        if xspeed >= 20:
            right = True
        elif xspeed <= -20:
            left = True


    # if y < 150:
    #     if x > x_limit:
    #         right = True
    #         left = False
    #     elif x < -x_limit:
    #         left = True
    #         right = False

    # x er positiv til højre for platformen og negativ til venstre for
    # xspeed er positiv gående mod højre og negativ gående mod venstre



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


env.close()
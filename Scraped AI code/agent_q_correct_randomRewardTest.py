import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from LunarLander import *
from collections import deque
import datetime

k_x = 1
k_y = 1
k_yspeed = 1
k_xspeed = 1

DISCOUNT = 0.99
BATCH_SIZE = 64
EPSILON = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
MEMORY_SIZE = 50_000
MIN_MEMORY_SIZE = 1000
TRAIN_AFTER = 5
SAVE_EVERY = 50
TRAIN_FROM_PRETRAINED = False
AGENT_PATH = './weights/agent272036.tar'
STABLE_AGENT_PATH = './weights/stable_agent272036.tar'


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


class DQN:
    def __init__(self):
        self.criterion = F.mse_loss
        self.model = Network()
        self.target_model = Network()
        self.target_update_counter = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

    def train(self, replay_memory):
        if len(replay_memory) < MIN_MEMORY_SIZE:
            return
        batch = random.sample(replay_memory, BATCH_SIZE)

        x = []
        y = []
        for sample in batch:
            state = sample[0]
            action = sample[1]
            reward = sample[2]
            model_input = sample[3]
            done = sample[4]

            current_q_values = self.model(model_input)
            x.append(current_q_values)
            next_state = DQN.get_next_state(state, action)
            next_model_input = DQN.get_model_input(next_state)
            next_q_values = self.target_model(next_model_input)

            if not done:
                max_next_q = torch.max(next_q_values)
                new_q = reward + DISCOUNT * max_next_q
            else:
                new_q = reward

            current_q_values[action] = new_q
            y.append(current_q_values)

        x = torch.stack(x)
        y = torch.stack(y)
        loss = self.criterion(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.target_update_counter == TRAIN_AFTER:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

    @staticmethod
    def get_reward(state, done):
        x, y, xspeed, yspeed = state

        if yspeed <= 20 and abs(x) <= 20 and abs(xspeed) <= 20 and y <= 0:
            print("VI VON")
            return 1000

        if done:
            return -5
        else:
            return 0

            return x_reward + y_reward + xspeed_reward + yspeed_reward

    @staticmethod
    def get_next_state(state, action):
        x, y, xspeed, yspeed = state
        boost = False
        left = False
        right = False

        if action == 1:
            left = True
        elif action == 2:
            boost = True
        elif action == 3:
            right = True
        elif action == 4:
            right = True
            boost = True
        elif action == 5:
            boost = True
            right = True

        if boost:
            yspeed -= 1.5
        else:
            yspeed += 1
        if left:
            xspeed += 2
        if right:
            xspeed -= 2

        y -= yspeed * .1
        x += xspeed * .1

        return x, y, xspeed, yspeed

    @staticmethod
    def update_replay_memory(state, action, reward, model_input, done):
        replay_memory.append((state, action, reward, model_input, done))

    @staticmethod
    def get_model_input(state):
        x, y, xspeed, yspeed = state
        return torch.tensor([x / 450, y / 600, xspeed / 200, yspeed / 200], dtype=torch.float32)

    @staticmethod
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
    time_created = datetime.datetime.now().timetuple()
    replay_memory = deque(maxlen=MEMORY_SIZE)
    if TRAIN_FROM_PRETRAINED:
        agent = DQN()
        agent.model.load_state_dict(torch.load(AGENT_PATH))
        agent.target_model.load_state_dict(torch.load(STABLE_AGENT_PATH))
    else:
        agent = DQN()

    env = LunarLander()
    env.reset()
    exit_program = False
    games_played = 0
    game_trained = 0

    while not exit_program:
        # if games_played > RENDER_AFTER:
        #     env.render()
        # env.render()

        (x, y, xspeed, yspeed), reward, done = env.step((boost, left, right))
        # print(env.rocket.fuel)
        state = (x, y, xspeed, yspeed)
        if env.rocket.fuel == 0:
            action = 0
        else:
            model_input = agent.get_model_input(state)

            if random.random() > EPSILON:
                q_values = agent.model(model_input)
                action = torch.argmax(q_values)
            else:
                action = random.randint(0, 5)

            reward = agent.get_reward(state, done)

            agent.update_replay_memory(state, action, reward, model_input, done)

            agent.train(replay_memory)

            if game_trained != games_played:
                game_trained = games_played
                if games_played % SAVE_EVERY == 0:
                    torch.save(agent.model.state_dict(),
                               f'weights/agent{time_created[2]}{time_created[3]}{time_created[4]}.tar')
                    torch.save(agent.target_model.state_dict(),
                               f'weights/stable_agent{time_created[2]}{time_created[3]}{time_created[4]}.tar')
                    print(f'models saved with id: {time_created[2]}{time_created[3]}{time_created[4]}')

        left, boost, right = agent.get_movement_variables(action)

        if done:
            agent.target_update_counter += 1
            games_played += 1
            if EPSILON > MIN_EPSILON:
                EPSILON *= EPSILON_DECAY
                EPSILON = max(MIN_EPSILON, EPSILON)
            if games_played % 10 == 0:
                print(games_played)
            env.reset()

    env.close()

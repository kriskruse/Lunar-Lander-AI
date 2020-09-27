import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from LunarLander import *
from collections import deque
import datetime


k_x = 100
k_y = 100
#k_yspeed = 25
#k_xspeed = 25


DISCOUNT = 0.99
BATCH_SIZE = 100
EPSILON = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
SAMPLE_SIZE = 1000
MEMORY_SIZE = 50_000
MIN_MEMORY_SIZE = 1000
TRAIN_AFTER = 5
RENDER_AFTER = 100
SAVE_EVERY = 100
TRAIN_FROM_PRETRAINED = False
AGENT_PATH = ''
STABLE_AGENT_PATH = ''


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 6)

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
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)

    def train_once(self, state, action, reward, model_input):
        predictions = self.model(model_input)

        next_state = self.get_next_state(state, action)
        next_model_input = self.get_model_input(next_state)
        q_values = self.model(model_input)

        # Denne skal predictes af target model
        new_q_values = self.model(next_model_input)

        if not done:
            max_new_q = torch.max(new_q_values)
            new_q = reward + DISCOUNT * max_new_q
        else:
            new_q = reward

        q_values[action] = new_q
        y = q_values

        loss = self.criterion(predictions, y)
        # print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, replay_memory):
        random.shuffle(replay_memory)
        sample = random.sample(replay_memory, SAMPLE_SIZE)
        ds = Dataset(sample)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, drop_last=True)

        for batch in dataloader:
            x, y = batch

            predictions = self.model(x)
            loss = self.criterion(predictions, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @staticmethod
    def get_reward(state, done):
        x, y, xspeed, yspeed = state

        if abs(yspeed) <= 20 and abs(x) <= 20 and abs(xspeed) <= 20 and abs(y):
            return 1000

        if done == True:
            return -5
        else:
            return -1 + 600/y*k_y + (400/max(abs(x),0.1))*k_x   # + 140/max(abs(yspeed),18)*k_yspeed + 140/max(abs(xspeed),18)*k_xspeed


        # max_distance = np.sqrt(400 ** 2 + 600 ** 2)
        # reward = (max_distance / np.sqrt(x ** 2 + y ** 2)) * k_distance + done*k_lost + 140/max(abs(yspeed),18)*k_speed + 140/max(abs(xspeed),18) * k_speed
        # reward = 1/max(abs(x), 0.0001)*10 + 1/max(abs(y), 0.0001)*10 + 140/max(abs(yspeed),18)*k_speed + 140/max(abs(xspeed),18) * k_speed
        # reward = abs(x) + abs(y) + 140/max(abs(yspeed),18)*k_speed + 140/max(abs(xspeed),18) * k_speed
        # return reward

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
        replay_memory.append((state, action, reward,  model_input, done))

    @staticmethod
    def get_model_input(state):
        x, y, xspeed, yspeed = state
        return torch.tensor([x/450, y / 600, xspeed / 200, yspeed / 200], dtype=torch.float32)

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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, replay_memory):
        self.memory = replay_memory

    def __getitem__(self, i):
        state = replay_memory[i][0]
        action = replay_memory[i][1]
        reward = replay_memory[i][2]
        model_input = replay_memory[i][3]
        done = replay_memory[i][4]

        current_q_values = agent.model(model_input)

        next_state = DQN.get_next_state(state, action)
        next_model_input = DQN.get_model_input(next_state)

        next_q_values = stable_agent.model(next_model_input)

        if not done:
            max_next_q = torch.max(next_q_values)
            new_q = reward + DISCOUNT * max_next_q
        else:
            new_q = reward

        current_q_values[action] = new_q

        return model_input, current_q_values

    def __len__(self):
        return len(replay_memory)


if __name__ == '__main__':
    time_created = datetime.datetime.now().timetuple()
    replay_memory = deque(maxlen=MEMORY_SIZE)
    if TRAIN_FROM_PRETRAINED:
        agent = DQN()
        agent.model.load_state_dict(torch.load(AGENT_PATH))
        stable_agent = DQN()
        stable_agent.model.load_state_dict(torch.load(STABLE_AGENT_PATH))
    else:
        agent = DQN()
        stable_agent = DQN()

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
        state = (x, y, xspeed, yspeed)
        model_input = agent.get_model_input(state)

        if random.random() > EPSILON:
            q_values = agent.model(model_input)
            action = torch.argmax(q_values)
        else:
            action = random.randint(0, 5)
            EPSILON *= EPSILON_DECAY
        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)

        reward = agent.get_reward(state, done)

        agent.update_replay_memory(state, action, reward, model_input, done)

        # make sure weights are being updated
        # a = list(model.model.parameters())[0].clone()
        agent.train_once(state, action, reward, model_input)
        # b = list(model.model.parameters())[0].clone()
        # print(torch.equal(a.data, b.data))

        # check for model being trainable
        # print(model.model.training)

        if len(replay_memory) > MIN_MEMORY_SIZE and games_played % TRAIN_AFTER == 0 and game_trained != games_played:
            game_trained = games_played
            agent.train(replay_memory)
            stable_agent.train(replay_memory)
            if games_played % 50 == 0:

                torch.save(agent.model.state_dict(), f'weights/agent{time_created[2]}{time_created[3]}{time_created[4]}.tar')
                torch.save(stable_agent.model.state_dict(), f'weights/stable_agent{time_created[2]}{time_created[3]}{time_created[4]}.tar')
                print(f'models saved with id: {time_created[2]}{time_created[3]}{time_created[4]}')

        left, boost, right = agent.get_movement_variables(action)

        if done:
            games_played += 1
            if games_played % 10 == 0:
                print(games_played)
            env.reset()

    env.close()

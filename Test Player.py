import Neural_Networks as nn
import numpy as np
import cv2
import retrowrapper


# File Paths
LOAD_PLAYER_PATH = "Pretrained_Player_Weights.h5"
GAME_PATH = "/path/to/Game Files"
STATE_PATH = "/path/to/ApeX/Game Files/Level1.state"


# Input Actions (Button Presses)
NO_ACTION = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
RIGHT_ACTION = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
LEFT_ACTION = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
UP_ACTION = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
DOWN_ACTION = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
ACTION_SET = [RIGHT_ACTION, LEFT_ACTION, UP_ACTION, DOWN_ACTION]


# Neural Network Parameters
WIDTH = 68
HEIGHT = 84
NUM_FRAMES = 2
STATE_SHAPE = (HEIGHT, WIDTH, NUM_FRAMES)
EXP_STATE_SHAPE = (1, HEIGHT, WIDTH, NUM_FRAMES)
NUM_ACTIONS = 4
NUM_DENSE_UNITS = 256
L2_REG = 5e-5
LEARNING_RATE = 1e-5
CLIP_NORM = 50.0
HUBER_DELTA = 1.0


# Player Parameters
EPSILON = 0
REWARD_NORMALIZATION = 50.0
FRAME_SKIP = 4
FRAMES_PER_STEP = NUM_FRAMES * FRAME_SKIP
DISCOUNT = 0.993
RENDER = True


def format_state(state):
    state = state[14:218, 6:162, :]
    state = cv2.resize(state, (WIDTH, HEIGHT), cv2.INTER_CUBIC)
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    return state / 255.0


def get_reward(info, past_score):
    return (info['score'] - past_score) / REWARD_NORMALIZATION


def get_done(info, reward):
    if info['level'] > 0:
        return True, reward + 0.5
    elif info['lives'] < 3 or info["pacmanGraphic"] > 16:
        return True, reward - 0.5
    else:
        return False, reward


class Player:
    def __init__(self):
        # How likely it is to explore
        self.epsilon = EPSILON

        # Initialize Neural Network
        self.player = nn.create_player_model(state_shape=STATE_SHAPE,
                                             num_actions=NUM_ACTIONS,
                                             num_dense_units=NUM_DENSE_UNITS,
                                             l2_regularizer=L2_REG,
                                             learning_rate=LEARNING_RATE,
                                             clipnorm=CLIP_NORM,
                                             huber_delta=HUBER_DELTA)
        self.player.load_weights(LOAD_PLAYER_PATH)

        # Initialize Game Environment
        self.env = retrowrapper.RetroWrapper(game=GAME_PATH,
                                             state=STATE_PATH)

    # Custom reset function that skips and stacks frames
    def env_reset(self):
        full_state = np.zeros(STATE_SHAPE)
        state = self.env.reset()
        if RENDER:
            self.env.render()
        full_state[:, :, 0] = format_state(state)
        n = 1
        for i in range(1, FRAMES_PER_STEP):
            state, _, _, _ = self.env.step(NO_ACTION)
            if i % FRAME_SKIP == 0:
                full_state[:, :, n] = format_state(state)
                n += 1
            if RENDER:
                self.env.render()
        return full_state

    # Custom step function that skips and stacks frames
    def env_step(self, action):
        full_state = np.zeros(STATE_SHAPE)
        n = 0
        for i in range(NUM_FRAMES * FRAME_SKIP):
            state, _, _, info = self.env.step(ACTION_SET[action])
            if i % FRAME_SKIP == 0:
                full_state[:, :, n] = format_state(state)
                n += 1
            if RENDER:
                self.env.render()
        return full_state, info

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            q_values = self.player.predict(np.expand_dims(state, axis=0)).squeeze()
            print(q_values)
            action = np.argmax(q_values)
        return action

    # Collects training data
    def play_game(self):
        past_score = 0
        state = self.env_reset()
        total_steps = 0
        total_reward = 0
        while True:
            action = self.choose_action(state)
            next_state, info = self.env_step(action)
            reward = get_reward(info, past_score)
            done, reward = get_done(info, reward)
            total_steps += 1
            total_reward += reward
            if done:
                break
            state = next_state
            past_score = info['score']


if __name__ == '__main__':
    player = Player()
    player.play_game()
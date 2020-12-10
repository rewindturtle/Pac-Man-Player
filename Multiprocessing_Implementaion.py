import Neural_Networks as nn
import numpy as np
import cv2
import pandas as pd
import retrowrapper
import _pickle as pkl
import multiprocessing as mp
from multiprocessing.managers import BaseManager as BM
import time
import os


# Debugging
RENDER = True
LOAD_WEIGHTS = False
PRINT_STATUS = True
USE_EXPERT_DATA = False
USE_TRAINING_WEIGHTS = True


# File Paths
GAME_PATH = "C:/Users/natha/PycharmProjects/ApeX/Game Files"
STATE_PATH = "C:/Users/natha/PycharmProjects/ApeX/Game Files/Level1.state"
EXPERT_FOLDER = "Expert Data/"
EXPERT_STATES_PATH = EXPERT_FOLDER + "Expert_States.pkl"
EXPERT_ACTIONS_PATH = EXPERT_FOLDER + "Expert_Actions.pkl"
EXPERT_REWARDS_PATH = EXPERT_FOLDER + "Expert_Rewards.pkl"
EXPERT_DONES_PATH = EXPERT_FOLDER + "Expert_Dones.pkl"
LOAD_WEIGHTS_PATH = "Weights_89000.h5"
SAVE_MODEL_PATH = "Models/Model"
SAVE_WEIGHTS_PATH = "Weights/Weights"
CSV_PATH = 'Training_Data.csv'
COLUMNS = ['Duration',
           'Number of Games Played',
           'Number of Training Episodes',
           'Number of Samples Generated',
           'Total Episode Reward',
           'Episode Score',
           'Episode Steps',
           'Epsilon']


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
NUM_ACTIONS = len(ACTION_SET)
NUM_DENSE_UNITS = 256
L2_REG = 5e-5
LEARNING_RATE = 1e-5
CLIP_NORM = 50.0
HUBER_DELTA = 1.0


# Players and Learners
NUM_PLAYERS = 2
MAX_EPSILON = 0.04
MIN_EPSILON = 2e-3
FRAME_SKIP = 4
FRAMES_PER_STEP = NUM_FRAMES * FRAME_SKIP
DISCOUNT = 0.993
REWARD_NORMALIZATION = 5.0


# Learners
NUM_LEARNERS = 1
MAX_MEMORY_SIZE = int(1e6)
MAX_TRAINING_EPISODES = int(1e5)
PER_EPSILON = 1e-10
PER_ALPHA = 0.4
PER_BETA = 0.6
BATCH_SIZE = 32
WARM_UP = 5 * BATCH_SIZE
N_STEP = 10


# Expert Imitation Parameters
MARGIN = 1
EMPTY_IS_EXPERT = np.zeros((BATCH_SIZE, 1))
EMPTY_MARGIN = MARGIN * np.ones((BATCH_SIZE, NUM_ACTIONS))
EMPTY_EXPERT_ACTIONS = np.zeros((BATCH_SIZE, NUM_ACTIONS))
EMPTY_LOSS = np.zeros((BATCH_SIZE, 1))
ROWS = np.arange(BATCH_SIZE).astype(int)
EXPERT_FRACTION = 0.25
EXPERT_BATCH_SIZE = int(EXPERT_FRACTION * BATCH_SIZE)
NORM_BATCH_SIZE = BATCH_SIZE - EXPERT_BATCH_SIZE


# Frequencies
UPDATE_TARGET_FREQUENCY = 2500
SAVE_FREQ = 1000
UPDATE_CSV_FREQ = 5


# Expert Data
if USE_EXPERT_DATA:
    with open(EXPERT_STATES_PATH, 'rb') as handle:
        EX_STATES = np.array(pkl.load(handle))
    with open(EXPERT_ACTIONS_PATH, 'rb') as handle:
        EX_ACTIONS = np.array(pkl.load(handle)).astype(int)
    with open(EXPERT_REWARDS_PATH, 'rb') as handle:
        EX_REWARDS = np.array(pkl.load(handle))
    with open(EXPERT_DONES_PATH, 'rb') as handle:
        EX_DONES = np.array(pkl.load(handle))


def format_state(state):
    state = state[14:218, 6:162, :]
    state = cv2.resize(state, (WIDTH, HEIGHT), cv2.INTER_CUBIC)
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    return state / 255.0


def get_reward(info, past_score):
    return (info['score'] - past_score) / REWARD_NORMALIZATION


def get_done(info, reward):
    if info['level'] > 0:
        return True, reward + 5
    elif info['lives'] < 3 or info["pacmanGraphic"] > 16:
        return True, reward
    else:
        return False, reward


class Global_Agent:
    def __init__(self):
        # Counters
        self.start_time = time.process_time()
        self.num_training_episodes = 0
        self.num_games_played = 0
        self.num_samples_generated = 0
        self.training_data = []
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(CSV_PATH, mode='w', header=True, index=False)

        # Threading Locks
        self.global_lock = mp.Lock()
        self.memory_lock = mp.Lock()
        self.learner_lock = mp.Lock()
        self.learner_lock.acquire()
        self.is_learner_locked = True
        if USE_EXPERT_DATA:
            self.expert_lock = mp.Lock()
        if PRINT_STATUS:
            self.display_lock = mp.Lock()

        # Initialize Neural Network
        self.global_agent = nn.create_global_model(state_shape=STATE_SHAPE,
                                                   num_actions=NUM_ACTIONS,
                                                   num_dense_units=NUM_DENSE_UNITS,
                                                   l2_regularizer=L2_REG,
                                                   learning_rate=LEARNING_RATE,
                                                   huber_delta=HUBER_DELTA,
                                                   clipnorm=CLIP_NORM)
        if LOAD_WEIGHTS:
            self.global_agent.load_weights(LOAD_WEIGHTS_PATH)
        if PRINT_STATUS:
            print(self.global_agent.summary())
        self.global_target = nn.create_global_model(state_shape=STATE_SHAPE,
                                                    num_actions=NUM_ACTIONS,
                                                    num_dense_units=NUM_DENSE_UNITS,
                                                    l2_regularizer=L2_REG,
                                                    learning_rate=LEARNING_RATE,
                                                    huber_delta=HUBER_DELTA,
                                                    clipnorm=CLIP_NORM)
        self.update_target()

        self.envs = []
        self.players = []
        for i in range(NUM_PLAYERS):
            self.envs.append(retrowrapper.RetroWrapper(game=GAME_PATH,
                                                       state=STATE_PATH))
            self.players.append(nn.create_player_model(state_shape=STATE_SHAPE,
                                                       num_actions=NUM_ACTIONS,
                                                       num_dense_units=NUM_DENSE_UNITS,
                                                       l2_regularizer=L2_REG,
                                                       learning_rate=LEARNING_RATE,
                                                       clipnorm=CLIP_NORM,
                                                       huber_delta=HUBER_DELTA))
            self.players[i].set_weights(self.global_agent.get_weights())

        # Initialize Memory
        self.states = np.empty(0)
        self.actions = np.empty(0)
        self.rewards = np.empty(0)
        self.dones = np.empty(0)
        self.td = np.empty(0)
        if USE_EXPERT_DATA:
            self.ex_td = np.array(len(EX_ACTIONS) * [10])

    # Copies weights and biases of the global agent to the target agent
    def update_target(self):
        self.global_target.set_weights(self.global_agent.get_weights())
        if PRINT_STATUS:
            self.display_lock.acquire()
            print("________________________________________________________________")
            print("Target Updated")
            self.display_lock.release()

    # Saves current tensorflow model and weights
    def save_model(self):
        self.global_agent.save("{}_{}.h5".format(SAVE_MODEL_PATH, self.num_training_episodes))
        self.global_agent.save_weights("{}_{}.h5".format(SAVE_WEIGHTS_PATH, self.num_training_episodes))
        if PRINT_STATUS:
            self.display_lock.acquire()
            print("________________________________________________________________")
            print("Model Saved")
            self.display_lock.release()

    def update_data_frame(self):
        df = pd.DataFrame(self.training_data, columns=COLUMNS)
        df.to_csv(CSV_PATH, mode='a', header=False, index=False)
        self.training_data = []
        if PRINT_STATUS:
            self.display_lock.acquire()
            print("________________________________________________________________")
            print("Training Data CSV Updated")
            self.display_lock.release()

    def update_player_weights(self, player_num):
        self.global_lock.acquire()
        self.players[player_num].set_weights(self.global_agent.get_weights())
        self.global_lock.release()

    # Custom reset function that skips and stacks frames
    def env_reset(self, player_num):
        full_state = np.zeros(STATE_SHAPE)
        state = self.envs[player_num].reset()
        if RENDER:
            self.envs[player_num].render()
        full_state[:, :, 0] = format_state(state)
        n = 1
        for i in range(1, FRAMES_PER_STEP):
            state, _, _, _ = self.envs[player_num].step(NO_ACTION)
            if i % FRAME_SKIP == 0:
                full_state[:, :, n] = format_state(state)
                n += 1
            if RENDER:
                self.envs[player_num].render()
        return full_state

    # Custom step function that skips and stacks frames
    def env_step(self, player_num, action):
        full_state = np.zeros(STATE_SHAPE)
        n = 0
        for i in range(NUM_FRAMES * FRAME_SKIP):
            state, _, _, info = self.envs[player_num].step(ACTION_SET[action])
            if i % FRAME_SKIP == 0:
                full_state[:, :, n] = format_state(state)
                n += 1
            if RENDER:
                self.envs[player_num].render()
        return full_state, info

    def player_predict(self, player_num, state):
        return self.players[player_num].predict(state).squeeze()

    def acquire_display_lock(self):
        self.display_lock.acquire()

    def release_display_lock(self):
        self.display_lock.release()

    def acquire_memory_lock(self):
        self.memory_lock.acquire()

    def release_memory_lock(self):
        self.memory_lock.release()

    def acquire_learner_lock(self):
        self.learner_lock.acquire()

    def release_learner_lock(self):
        self.learner_lock.release()

    def get_num_training_episodes(self):
        return self.num_training_episodes

    def update_memory(self, total_reward, score, total_steps, epsilon,
                      states, actions, rewards, dones, td):
        self.memory_lock.acquire()
        self.num_games_played += 1
        self.num_samples_generated += len(td)

        self.training_data.append([time.process_time() - self.start_time,
                                   self.num_games_played,
                                   self.num_training_episodes,
                                   self.num_samples_generated,
                                   total_reward,
                                   score,
                                   total_steps,
                                   epsilon])
        if len(self.states) == 0:
            self.states = states
        else:
            self.states = np.concatenate((self.states, states), axis=0)
        self.actions = np.append(self.actions, actions)
        self.rewards = np.append(self.rewards, rewards)
        self.dones = np.append(self.dones, dones)
        self.td = np.append(self.td, td)

        td_len = len(self.td)
        if td_len > MAX_MEMORY_SIZE:
            diff = td_len - MAX_MEMORY_SIZE
            self.states = self.states[diff:]
            self.actions = self.actions[diff:]
            self.rewards = self.rewards[diff:]
            self.dones = self.dones[diff:]
            self.td = self.td[diff:]

        if self.is_learner_locked and td_len > WARM_UP:
            self.learner_lock.release()
            self.is_learner_locked = False

        if len(self.training_data) % UPDATE_CSV_FREQ == 0:
            self.update_data_frame()

        self.memory_lock.release()

    def get_memory_batch(self):
        self.memory_lock.acquire()
        len_memory = len(self.td)
        sample_ref = self.num_samples_generated
        td_prob = self.td / np.sum(self.td)
        if USE_EXPERT_DATA:
            batch_indices = np.random.choice(len_memory,
                                             NORM_BATCH_SIZE,
                                             replace=False,
                                             p=td_prob)
        else:
            batch_indices = np.random.choice(len_memory,
                                             BATCH_SIZE,
                                             replace=False,
                                             p=td_prob)
        current_states = self.states[batch_indices]
        actions = self.actions[batch_indices].astype(int)
        one_rewards = self.rewards[batch_indices]
        one_dones = self.dones[batch_indices]
        n_rewards = np.empty(0)
        n_dones = np.empty(0)
        one_next_states = np.empty(0)
        n_next_states = np.empty(0)
        exponents = np.empty(0)
        for idx in batch_indices:
            r = 0
            for n in range(N_STEP):
                r += (DISCOUNT ** n) * self.rewards[idx + n]
                if self.dones[idx + n]:
                    break
            n_rewards = np.append(n_rewards, r)
            n_dones = np.append(n_dones, self.dones[idx + n])
            exponents = np.append(exponents, n + 1)
            if idx + 1 == len_memory:
                if len(one_next_states) == 0:
                    one_next_states = np.zeros(EXP_STATE_SHAPE)
                else:
                    one_next_states = np.concatenate((one_next_states,
                                                      np.zeros(EXP_STATE_SHAPE)),
                                                     axis=0)
            else:
                if len(one_next_states) == 0:
                    one_next_states = np.expand_dims(self.states[idx + 1], axis=0)
                else:
                    one_next_states = np.concatenate((one_next_states,
                                                      np.expand_dims(self.states[idx + 1], axis=0)),
                                                     axis=0)
            if idx + n + 1 == len_memory:
                if len(n_next_states) == 0:
                    n_next_states = np.zeros(EXP_STATE_SHAPE)
                else:
                    n_next_states = np.concatenate((n_next_states,
                                                    np.zeros(EXP_STATE_SHAPE)),
                                                   axis=0)
            else:
                if len(n_next_states) == 0:
                    n_next_states = np.expand_dims(self.states[idx + n + 1], axis=0)
                else:
                    n_next_states = np.concatenate((n_next_states,
                                                    np.expand_dims(self.states[idx + n + 1], axis=0)),
                                                   axis=0)
        self.memory_lock.release()
        return current_states, actions, one_rewards, one_dones, n_rewards, n_dones, \
               one_next_states, n_next_states, exponents, sample_ref, td_prob, batch_indices

    def get_expert_batch(self):
        self.expert_lock.acquire()
        num_ex_samples = len(self.ex_td)
        ex_td_prob = self.ex_td / np.sum(self.ex_td)
        ex_batch_indices = np.random.choice(num_ex_samples,
                                            EXPERT_BATCH_SIZE,
                                            replace=False,
                                            p=ex_td_prob)
        self.expert_lock.release()
        ex_current_states = EX_STATES[ex_batch_indices]
        ex_actions = EX_ACTIONS[ex_batch_indices]
        ex_one_rewards = EX_REWARDS[ex_batch_indices]
        ex_one_dones = EX_DONES[ex_batch_indices]
        ex_n_rewards = np.empty(0)
        ex_n_dones = np.empty(0)
        ex_one_next_states = np.empty(0)
        ex_n_next_states = np.empty(0)
        ex_exponents = np.empty(0)
        for idx in ex_batch_indices:
            r = 0
            for n in range(N_STEP):
                r += (DISCOUNT ** n) * EX_REWARDS[idx + n]
                if EX_DONES[idx + n]:
                    break
            ex_n_rewards = np.append(ex_n_rewards, r)
            ex_n_dones = np.append(ex_n_dones, EX_DONES[idx + n])
            ex_exponents = np.append(ex_exponents, n + 1)
            if idx + 1 == num_ex_samples:
                if len(ex_one_next_states) == 0:
                    ex_one_next_states = np.zeros(EXP_STATE_SHAPE)
                else:
                    ex_one_next_states = np.concatenate((ex_one_next_states,
                                                         np.zeros(EXP_STATE_SHAPE)),
                                                        axis=0)
            else:
                if len(ex_one_next_states) == 0:
                    ex_one_next_states = np.expand_dims(EX_STATES[idx + 1], axis=0)
                else:
                    ex_one_next_states = np.concatenate((ex_one_next_states,
                                                         np.expand_dims(EX_STATES[idx + 1], axis=0)),
                                                        axis=0)
            if idx + n + 1 == num_ex_samples:
                if len(ex_n_next_states) == 0:
                    ex_n_next_states = np.zeros(EXP_STATE_SHAPE)
                else:
                    ex_n_next_states = np.concatenate((ex_n_next_states,
                                                       np.zeros(EXP_STATE_SHAPE)),
                                                      axis=0)
            else:
                if len(ex_n_next_states) == 0:
                    ex_n_next_states = np.expand_dims(EX_STATES[idx + n + 1], axis=0)
                else:
                    ex_n_next_states = np.concatenate((ex_n_next_states,
                                                       np.expand_dims(EX_STATES[idx + n + 1], axis=0)),
                                                      axis=0)
        return ex_current_states, ex_actions, ex_one_rewards, ex_one_dones, ex_n_rewards, ex_n_dones,\
               ex_one_next_states, ex_n_next_states, ex_exponents, ex_td_prob, ex_batch_indices

    def get_training_qs(self, current_states, one_next_states, n_next_states):
        self.global_lock.acquire()
        cur_q, _, _ = self.global_agent.predict([current_states,
                                                 current_states,
                                                 EMPTY_IS_EXPERT,
                                                 EMPTY_MARGIN,
                                                 EMPTY_EXPERT_ACTIONS])
        one_next_q, n_next_q, _ = self.global_agent.predict([one_next_states,
                                                             n_next_states,
                                                             EMPTY_IS_EXPERT,
                                                             EMPTY_MARGIN,
                                                             EMPTY_EXPERT_ACTIONS])
        one_target_q, n_target_q, _ = self.global_target.predict([one_next_states,
                                                                  n_next_states,
                                                                  EMPTY_IS_EXPERT,
                                                                  EMPTY_MARGIN,
                                                                  EMPTY_EXPERT_ACTIONS])
        self.global_lock.release()
        return cur_q, one_next_q, n_next_q, one_target_q, n_target_q

    def train_global_network(self, current_states, is_expert, margins, expert_actions,
                             training_one_q, training_n_q, training_weights, one_next_states):
        self.global_lock.acquire()
        self.global_agent.fit([current_states,
                               current_states,
                               is_expert,
                               margins,
                               expert_actions],
                              [training_one_q,
                               training_n_q,
                               EMPTY_LOSS],
                              batch_size=BATCH_SIZE,
                              verbose=0,
                              sample_weight=[training_weights,
                                             training_weights,
                                             training_weights])
        q, nq, im_loss = self.global_agent.predict([current_states,
                                                    one_next_states,
                                                    is_expert,
                                                    margins,
                                                    expert_actions])
        self.num_training_episodes += 1
        if self.num_training_episodes % UPDATE_TARGET_FREQUENCY == 0:
            self.update_target()
        if self.num_training_episodes % SAVE_FREQ == 0:
            self.save_model()
        self.global_lock.release()
        return q, nq, im_loss

    def update_expert_td(self, ex_batch_indices, ex_td):
        self.expert_lock.acquire()
        self.ex_td[ex_batch_indices] = ex_td
        self.expert_lock.release()

    def update_memory_td(self, sample_ref, batch_indices, td):
        self.memory_lock.acquire()
        if USE_EXPERT_DATA:

            if self.num_samples_generated > MAX_MEMORY_SIZE:
                diff = self.num_samples_generated - sample_ref
                shifted_batch_indices = np.max(batch_indices - diff, -1)
                self.td[shifted_batch_indices] = np.where(shifted_batch_indices == -1,
                                                          self.td[shifted_batch_indices],
                                                          td[:NORM_BATCH_SIZE])
            else:
                self.td[batch_indices] = td[:NORM_BATCH_SIZE]
        else:
            if self.num_samples_generated > MAX_MEMORY_SIZE:
                diff = self.num_samples_generated - sample_ref
                shifted_batch_indices = np.max(batch_indices - diff, -1)
                self.td[shifted_batch_indices] = np.where(shifted_batch_indices == -1,
                                                          self.td[shifted_batch_indices],
                                                          td)
            else:
                self.td[batch_indices] = td
        self.memory_lock.release()


class Player:
    def __init__(self, player_num):
        # Counters
        self.player_num = player_num
        self.num_games_played = 0

        # How likely it is to explore
        if NUM_PLAYERS == 1:
            self.epsilon = MAX_EPSILON
        else:
            alpha = np.log(MIN_EPSILON) + player_num * (np.log(MAX_EPSILON) - np.log(MIN_EPSILON)) / (NUM_PLAYERS - 1)
            self.epsilon = np.exp(alpha)

    def choose_action(self, q_values):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action = np.argmax(q_values)
        return action, q_values[action], np.max(q_values)

    # Collects training data
    def play_game(self, global_agent):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        if PRINT_STATUS:
            global_agent.acquire_display_lock()
            print("Player {} has started playing!".format(self.player_num))
            global_agent.release_display_lock()
        while True:
            if global_agent.get_num_training_episodes() > MAX_TRAINING_EPISODES:
                break
            states = np.empty(0)
            actions = np.empty(0)
            rewards = np.empty(0)
            dones = np.empty(0)
            chosen_qs = np.empty(0)
            max_qs = np.empty(0)
            past_score = 0
            state = global_agent.env_reset(self.player_num)
            total_steps = 0
            total_reward = 0
            while True:
                q_values = global_agent.player_predict(self.player_num, np.expand_dims(state, axis=0))
                action, chosen_q, max_q = self.choose_action(q_values)
                next_state, info = global_agent.env_step(self.player_num, action)
                reward = get_reward(info, past_score)
                done, reward = get_done(info, reward)

                if len(states) == 0:
                    states = np.expand_dims(state, axis=0)
                else:
                    states = np.concatenate((states, np.expand_dims(state, axis=0)), axis=0)
                actions = np.append(actions, action)
                rewards = np.append(rewards, reward)
                dones = np.append(dones, done)
                chosen_qs = np.append(chosen_qs, chosen_q)
                max_qs = np.append(max_qs, max_q)

                total_steps += 1
                total_reward += reward
                if done:
                    break
                state = next_state
                past_score = info['score']

            self.num_games_played += 1
            max_qs = np.append(max_qs[1:], 0)
            q = rewards + DISCOUNT * max_qs - chosen_qs
            td = (abs(q) + PER_EPSILON) ** PER_ALPHA

            global_agent.update_memory(total_reward,
                                       10 * info['score'],
                                       total_steps,
                                       self.epsilon,
                                       states,
                                       actions,
                                       rewards,
                                       dones,
                                       td)
            global_agent.update_player_weights(self.player_num)


class Learner:
    def __init__(self, learner_num):
        self.learner_num = learner_num

    def learn(self, global_agent):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        if PRINT_STATUS:
            global_agent.acquire_display_lock()
            print("Learner {} has started learning!".format(self.learner_num))
            global_agent.release_display_lock()
        while True:
            if global_agent.get_num_training_episodes() > MAX_TRAINING_EPISODES:
                break
            start_time = time.time()
            current_states, actions, one_rewards, one_dones, n_rewards, n_dones, \
            one_next_states, n_next_states, exponents,\
            sample_ref, td_prob, batch_indices = global_agent.get_memory_batch()

            if USE_EXPERT_DATA:
                ex_current_states, ex_actions, ex_one_rewards, ex_one_dones, ex_n_rewards, ex_n_dones, \
                ex_one_next_states, ex_n_next_states, ex_exponents,\
                ex_td_prob, ex_batch_indices = global_agent.get_expert_batch()


                current_states = np.concatenate((current_states, ex_current_states), axis=0)
                one_next_states = np.concatenate((one_next_states, ex_one_next_states), axis=0)
                n_next_states = np.concatenate((n_next_states, ex_n_next_states), axis=0)
                actions = np.append(actions, ex_actions)
                one_rewards = np.append(one_rewards, ex_one_rewards)
                n_rewards = np.append(n_rewards, ex_n_rewards)
                one_dones = np.append(one_dones, ex_one_dones)
                n_dones = np.append(n_dones, ex_n_dones)
                exponents = np.append(exponents, ex_exponents)

            if USE_TRAINING_WEIGHTS:
                training_weights = td_prob[batch_indices]
                if USE_EXPERT_DATA:
                    training_weights = np.append((1 - EXPERT_FRACTION) * training_weights,
                                                 EXPERT_FRACTION * ex_td_prob[ex_batch_indices])
                training_weights = training_weights ** -PER_BETA
                training_weights = training_weights / np.max(training_weights)
            else:
                training_weights = np.ones(BATCH_SIZE)

            cur_q, one_next_q, n_next_q, \
            one_target_q, n_target_q = global_agent.get_training_qs(current_states,
                                                                    one_next_states,
                                                                    n_next_states)
            training_one_q = cur_q.copy()
            training_n_q = cur_q.copy()

            if USE_EXPERT_DATA:
                is_expert = np.append(np.zeros(NORM_BATCH_SIZE), np.ones(EXPERT_BATCH_SIZE))
                expert_actions = np.zeros((BATCH_SIZE, NUM_ACTIONS))
                expert_actions[ROWS, actions] = 1
                margins = MARGIN * (1 - expert_actions)
            else:
                is_expert = EMPTY_IS_EXPERT
                expert_actions = EMPTY_EXPERT_ACTIONS
                margins = EMPTY_MARGIN

            max_a = np.argmax(one_next_q, axis=1)
            training_one_q[ROWS, actions] = one_rewards + \
                                            DISCOUNT * (1 - one_dones) * one_target_q[ROWS, max_a]

            n_max_a = np.argmax(n_next_q, axis=1)
            training_n_q[ROWS, actions] = n_rewards + \
                                          (DISCOUNT ** exponents) * (1 - n_dones) * n_target_q[ROWS, n_max_a]

            q, nq, im_loss = global_agent.train_global_network(current_states,
                                                               is_expert,
                                                               margins,
                                                               expert_actions,
                                                               training_one_q,
                                                               training_n_q,
                                                               training_weights,
                                                               one_next_states)

            chosen_q = q[ROWS, actions]
            r = one_rewards + DISCOUNT * (1 - one_dones) * np.max(nq, axis=1) - chosen_q
            td = (abs(r) + PER_EPSILON) ** PER_ALPHA

            if USE_EXPERT_DATA:
                global_agent.update_expert_td(ex_batch_indices, td[NORM_BATCH_SIZE:])

            global_agent.update_memory_td(sample_ref, batch_indices, td)
            print(round(time.time() - start_time, 3))


if __name__ == '__main__':
    BM.register('Global_Agent', Global_Agent)
    manager = BM()
    manager.start()
    inst = manager.Global_Agent()
    players = []
    learners = []
    for i in range(NUM_PLAYERS):
        players.append(Player(i))
    for j in range(NUM_LEARNERS):
        learners.append(Learner(j))
    player_processes = []
    for player in players:
        p = mp.Process(target=player.play_game, args=[inst])
        p.start()
        player_processes.append(p)
        time.sleep(0.1)
    inst.acquire_learner_lock()
    learner_processes = []
    for learner in learners:
        l = mp.Process(target=learner.learn, args=[inst])
        l.start()
        learner_processes.append(l)
        time.sleep(0.1)
    for process in player_processes:
        process.join()
    for thread in learner_processes:
        thread.join()
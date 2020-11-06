import retrowrapper
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
import time
from collections import deque
import os
from matplotlib import pyplot as plt
import seaborn as sns
import multiprocessing as mp
from multiprocessing.managers import BaseManager
try:
    import _pickle as pickle
except:
    import pickle
sns.set()


# File Paths
GAME_PATH = "Game Files"
STATE_PATH = "Game Files/Level1.state"
EXPERT_DATA_PATH = "Complete_Expert_Data.pkl"
LOAD_WEIGHTS_PATH = "Expert_Weights_Global.h5"
SAVE_MODEL_PATH = "Models/Model"
SAVE_WEIGHTS_PATH = "Weights/Weights"

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
FRAME_SKIP = 4
STATE_SHAPE = (HEIGHT, WIDTH, NUM_FRAMES)
NUM_ACTIONS = len(ACTION_SET)
DENSE_UNITS = 512
L2_REG = 5e-5

# Hyper-Parameters
ADAM_LR = 1e-4
CLIP_NORM = 40
DISCOUNT = 0.99
REWARD_NORMALIZATION = 5
PER_EPSILON = 1e-5
PER_ALPHA = 0.4
PER_BETA = 0.7
N_STEP = 10

# Training Parameters
NUM_PLAYERS = 2
NUM_LEARNERS = 1
MAX_MEMORY = int(1e6)
MAX_TRAINING_EPISODES = int(1e5)
MAX_EPSILON = 0.04
MIN_EPSILON = 0.001
BATCH_SIZE = 512
WARM_UP = 100 * BATCH_SIZE
TARGET_UPDATE_FREQ = 1000
SAVE_FREQ = 1000
PLOT_FREQ = 10

# Expert Imitation Parameters
MARGIN = 1
EMPTY_IS_EXPERT = np.zeros((BATCH_SIZE, 1))
EMPTY_MARGIN = MARGIN * np.ones((BATCH_SIZE, NUM_ACTIONS))
EMPTY_EXPERT_ACTIONS = np.zeros((BATCH_SIZE, NUM_ACTIONS))
EMPTY_LOSS = np.zeros((BATCH_SIZE, 1))
EXPERT_FRACTION = 0.25
NORM_FRACTION = 1 - EXPERT_FRACTION
EXPERT_BATCH_SIZE = int(EXPERT_FRACTION * BATCH_SIZE)
NORM_BATCH_SIZE = BATCH_SIZE - EXPERT_BATCH_SIZE

# Debugging
RENDER = True
LOAD_WEIGHTS = False
USE_EXPERT = True
USE_TRAINING_WEIGHTS = True


def format_state(state):
    state = state[14:218, 6:162, :]
    state = cv2.resize(state, (WIDTH, HEIGHT), cv2.INTER_CUBIC)
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    return state / 255


def get_reward(info, past_score):
    return (info['score'] - past_score) / REWARD_NORMALIZATION


def get_done(info, reward):
    if info['level'] > 0:
        return True, reward + 5
    elif info['lives'] < 3:
        return True, reward
    elif info["pacmanGraphic"] > 16:
        return True, reward
    else:
        return False, reward


def get_action(action):
    return ACTION_SET[action]


def dueling_output(input_layer):
    state_layer = input_layer[0]
    action_layer = input_layer[1]
    return state_layer + action_layer - K.mean(action_layer, axis=1, keepdims=True)


def expert_output(margin_input):
    is_expert = margin_input[0]  # 1 if expert action, 0 otherwise
    expert_margin = margin_input[1]  # Defines the l_loss
    expert_action = margin_input[2]  # one-hot vector of the action take by the expert
    pred_q_values = margin_input[3]  # the q values predicted by the current nn
    expert_q = K.sum(pred_q_values * expert_action, axis=1, keepdims=True)
    relu_values = K.relu(pred_q_values + expert_margin - expert_q)
    mean_values = K.mean(relu_values, axis=1, keepdims=True)
    return is_expert * mean_values


def dqn_loss(y_true, y_pred):
    loss = K.sqrt(K.square((y_true - y_pred)) + 1) - 1
    return K.mean(loss)


def create_base_model():
    input_layer = tf.keras.layers.Input(shape=STATE_SHAPE,
                                        name="Input_Layer")
    conv_1 = tf.keras.layers.Conv2D(16,
                                    kernel_size=(8, 8),
                                    strides=4,
                                    kernel_initializer='he_uniform',
                                    input_shape=STATE_SHAPE,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                    bias_regularizer=tf.keras.regularizers.l2(L2_REG),
                                    name='Conv_Layer_1')(input_layer)
    conv_2 = tf.keras.layers.Conv2D(32,
                                    kernel_size=(4, 4),
                                    strides=2,
                                    kernel_initializer='he_uniform',
                                    input_shape=STATE_SHAPE,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                    bias_regularizer=tf.keras.regularizers.l2(L2_REG),
                                    name='Conv_Layer_2')(conv_1)
    conv_3 = tf.keras.layers.Conv2D(32,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    kernel_initializer='he_uniform',
                                    input_shape=STATE_SHAPE,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                    bias_regularizer=tf.keras.regularizers.l2(L2_REG),
                                    name='Conv_Layer_3')(conv_2)
    flat = tf.keras.layers.Flatten(name='Flat_Layer')(conv_3)
    state_dense = tf.keras.layers.Dense(DENSE_UNITS,
                                        activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                        bias_regularizer=tf.keras.regularizers.l2(L2_REG),
                                        kernel_initializer='he_uniform',
                                        name='State_Dense_Layer')(flat)
    state_value = tf.keras.layers.Dense(1,
                                        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                        bias_regularizer=tf.keras.regularizers.l2(L2_REG),
                                        name='State_Value_Layer')(state_dense)
    action_dense = tf.keras.layers.Dense(DENSE_UNITS,
                                         activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                         bias_regularizer=tf.keras.regularizers.l2(L2_REG),
                                         kernel_initializer='he_uniform',
                                         name='Action_Dense_Layer')(flat)
    action_value = tf.keras.layers.Dense(NUM_ACTIONS,
                                         activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                         bias_regularizer=tf.keras.regularizers.l2(L2_REG),
                                         kernel_initializer='he_uniform',
                                         name='Action_Value_Layer')(action_dense)
    output_layer = tf.keras.layers.Lambda(dueling_output,
                                          name='Output_Layer')([state_value, action_value])
    base_model = tf.keras.Model(input_layer,
                                output_layer,
                                name='Base_Model')
    return base_model


def create_player_model():
    model = create_base_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=ADAM_LR,
                                                     clipnorm=CLIP_NORM),
                  loss=dqn_loss)
    #model.make_predict_function()
    return model


def create_global_model():
    base_model = create_base_model()
    one_step_input = tf.keras.layers.Input(shape=STATE_SHAPE,
                                           name='One_Step_Input_Layer')
    n_step_input = tf.keras.layers.Input(shape=STATE_SHAPE,
                                         name='N_Step_Input_Layer')
    is_expert_input = tf.keras.layers.Input(shape=(1,),
                                            name='Is_Expert_Input')
    margin_input = tf.keras.layers.Input(shape=(NUM_ACTIONS,),
                                         name='Margin_Input')
    expert_action_input = tf.keras.layers.Input(shape=(NUM_ACTIONS,),
                                                name='Expert_Action_Input')
    one_step_output = base_model(one_step_input)
    n_step_output = base_model(n_step_input)
    margin_output = tf.keras.layers.Lambda(expert_output,
                                           name='Expert_Output')([is_expert_input,
                                                                  margin_input,
                                                                  expert_action_input,
                                                                  one_step_output])
    model = tf.keras.Model([one_step_input,
                            n_step_input,
                            is_expert_input,
                            margin_input,
                            expert_action_input],
                           [one_step_output,
                            n_step_output,
                            margin_output],
                           name='Global_Model')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=ADAM_LR,
                                                     ),
                  loss=[dqn_loss, dqn_loss, 'mae'])
    return model


class GlobalManager:
    def __init__(self):
        self.start_time = time.time()
        self.training_episodes = 0
        self.games_played = 0
        self.num_samples = 0

        self.global_lock = mp.Lock()
        self.memory_lock = mp.Lock()
        self.expert_lock = mp.Lock()
        self.display_lock = mp.Lock()
        self.warm_up_event = mp.Event()
        self.warm_up_event.clear()

        self.global_agent = create_global_model()
        if LOAD_WEIGHTS:
            self.global_agent.load_weights(LOAD_WEIGHTS_PATH)
        self.global_target = create_global_model()
        self.update_target()

        self.envs = []
        self.player_agents = []
        for i in range(NUM_PLAYERS):
            self.envs.append(retrowrapper.RetroWrapper(game=GAME_PATH,
                                                       state=STATE_PATH))
            player = create_player_model()
            player.set_weights(self.global_agent.get_weights())
            self.player_agents.append(player)
        print(self.player_agents[0].summary())

        self.memory = deque(maxlen=MAX_MEMORY)
        self.td_memory = deque(maxlen=MAX_MEMORY)
        if USE_EXPERT:
            with open(EXPERT_DATA_PATH, 'rb') as handle:
                self.expert_memory = pickle.load(handle)
                self.expert_td_memory = len(self.expert_memory) * [10]

        self.score_memory = []

    def update_target(self):
        self.global_target.set_weights(self.global_agent.get_weights())
        self.display_lock.acquire()
        print("________________________________________________________________")
        print("Target Updated")
        self.display_lock.release()

    def save_model(self):
        self.global_agent.save("{}_{}.h5".format(SAVE_MODEL_PATH, self.training_episodes))
        self.global_agent.save_weights("{}_{}.h5".format(SAVE_WEIGHTS_PATH, self.training_episodes))
        self.display_lock.acquire()
        print("________________________________________________________________")
        print("Model Saved")
        self.display_lock.release()

    def display_info(self, name, score, epsilon):
        seconds = time.time() - self.start_time
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        self.display_lock.acquire()
        print("________________________________________________________________")
        print("{}".format(name))
        print("Score: {}".format(score))
        print("Memory Length: {}".format(len(self.memory)))
        print("Samples Generated: {}".format(self.num_samples))
        print("Games Played: {}".format(self.games_played))
        print("Training Episodes: {}".format(self.training_episodes))
        print("Epsilon: {}".format(round(epsilon, 5)))
        print('Duration: {:d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds)))
        self.display_lock.release()

    def player_predict(self, player_num, state):
        return self.player_agents[player_num].predict(state).squeeze()

    def update_player_weights(self, player_num):
        self.global_lock.acquire()
        self.player_agents[player_num].set_weights(self.global_agent.get_weights())
        self.global_lock.release()

    def get_training_episodes(self):
        return self.training_episodes

    def env_reset(self, player_num):
        full_state = np.zeros(STATE_SHAPE)
        state = self.envs[player_num].reset()
        if RENDER:
            self.envs[player_num].render()
        full_state[:, :, 0] = format_state(state)
        n = 1
        for i in range(1, NUM_FRAMES * FRAME_SKIP):
            state, _, _, info = self.envs[player_num].step(NO_ACTION)
            if i % FRAME_SKIP == 0:
                full_state[:, :, n] = format_state(state)
                n += 1
            if RENDER:
                self.envs[player_num].render()
        return full_state

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

    def add_to_memory(self, episode_batch, td_errors, score, name, epsilon):
        self.memory_lock.acquire()
        self.games_played += 1
        self.score_memory.append(score)
        for i in range(len(td_errors)):
            self.memory.append(episode_batch[i])
            self.td_memory.append(td_errors[i])
        self.num_samples += len(td_errors)
        if self.num_samples > WARM_UP and not self.warm_up_event.is_set():
            self.warm_up_event.set()
        self.memory_lock.release()
        self.display_info(name, score, epsilon)
        if self.games_played % PLOT_FREQ == 0:
            self.plot_score()

    def acquire_memory_lock(self):
        self.memory_lock.acquire()

    def release_memory_lock(self):
        self.memory_lock.release()

    def acquire_expert_lock(self):
        self.expert_lock.acquire()

    def release_expert_lock(self):
        self.expert_lock.release()

    def acquire_display_lock(self):
        self.display_lock.acquire()

    def release_display_lock(self):
        self.display_lock.release()

    def wait_for_warm_up(self):
        self.warm_up_event.wait()
        self.display_lock.acquire()
        print("________________________________________________________________")
        print('Warm Up Complete!')
        self.display_lock.release()

    def get_num_samples(self):
        return self.num_samples

    def get_td_prob(self):
        return np.array(self.td_memory) / np.sum(self.td_memory)

    def get_expert_td_prob(self):
        return np.array(self.expert_td_memory) / np.sum(self.expert_td_memory)

    def get_batch(self, indices):
        current_states = []
        actions = []
        one_rewards = []
        n_rewards = []
        one_dones = []
        n_dones = []
        one_next_states = []
        n_next_states = []
        exponents = []
        for idx in indices:
            current_states.append(self.memory[idx][0])
            actions.append(self.memory[idx][1])
            one_rewards.append(self.memory[idx][2])
            one_dones.append(self.memory[idx][3])
            r = 0
            for n in range(N_STEP):
                r += (DISCOUNT ** n) * self.memory[idx + n][2]
                if self.memory[idx + n][3]:
                    break
            n_rewards.append(r)
            n_dones.append(self.memory[idx + n][3])
            if idx + 1 == len(self.memory):
                one_next_states.append(np.zeros(STATE_SHAPE))
            else:
                one_next_states.append(self.memory[idx + 1][0])
            if idx + n + 1 == len(self.memory):
                n_next_states.append(np.zeros(STATE_SHAPE))
            else:
                n_next_states.append(self.memory[idx + n + 1][0])
            exponents.append(n + 1)
        return current_states, actions, one_rewards, n_rewards, one_dones, n_dones, one_next_states, \
               n_next_states, exponents

    def get_expert_batch(self, indices):
        current_states = []
        actions = []
        one_rewards = []
        n_rewards = []
        one_dones = []
        n_dones = []
        one_next_states = []
        n_next_states = []
        exponents = []
        for idx in indices:
            current_states.append(self.expert_memory[idx][0])
            actions.append(self.expert_memory[idx][1])
            one_rewards.append(self.expert_memory[idx][2])
            one_dones.append(self.expert_memory[idx][3])
            r = 0
            for n in range(N_STEP):
                r += (DISCOUNT ** n) * self.expert_memory[idx + n][2]
                if self.expert_memory[idx + n][3]:
                    break
            n_rewards.append(r)
            n_dones.append(self.expert_memory[idx + n][3])
            if idx + 1 == len(self.expert_memory):
                one_next_states.append(np.zeros(STATE_SHAPE))
            else:
                one_next_states.append(self.expert_memory[idx + 1][0])
            if idx + n + 1 == len(self.expert_memory):
                n_next_states.append(np.zeros(STATE_SHAPE))
            else:
                n_next_states.append(self.expert_memory[idx + n + 1][0])
            exponents.append(n + 1)
        return current_states, actions, one_rewards, n_rewards, one_dones, n_dones, one_next_states, \
               n_next_states, exponents

    def global_predict(self, current_states, one_next_states, n_next_states):
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

    def norm_train(self, current_states, training_one_q, training_n_q, training_weights,
                   next_states, actions, rewards, dones, sample_ref, batch_indices):
        self.global_lock.acquire()
        self.global_agent.fit([current_states,
                               current_states,
                               EMPTY_IS_EXPERT,
                               EMPTY_MARGIN,
                               EMPTY_EXPERT_ACTIONS],
                              [training_one_q,
                               training_n_q,
                               EMPTY_LOSS],
                              batch_size=BATCH_SIZE,
                              verbose=0,
                              sample_weight=[training_weights,
                                             training_weights,
                                             training_weights])
        cur_q, next_q, _ = self.global_agent.predict([current_states,
                                                      next_states,
                                                      EMPTY_IS_EXPERT,
                                                      EMPTY_MARGIN,
                                                      EMPTY_EXPERT_ACTIONS])
        self.training_episodes += 1
        if self.training_episodes % TARGET_UPDATE_FREQ == 0:
            self.update_target()
        if self.training_episodes % SAVE_FREQ == 0:
            self.save_model()
        self.global_lock.release()
        td_errors = []
        for i in range(BATCH_SIZE):
            a = actions[i]
            if dones[i]:
                td = (abs(rewards[i] - cur_q[i, a]) + PER_EPSILON) ** PER_ALPHA
            else:
                td = (abs(rewards[i] + DISCOUNT * np.max(next_q[i]) - cur_q[i, a]) + PER_EPSILON) ** PER_ALPHA
            td_errors.append(td)
        self.memory_lock.acquire()
        for i in range(BATCH_SIZE):
            if self.num_samples <= MAX_MEMORY:
                self.td_memory[batch_indices[i]] = td_errors[i]
            else:
                idx = batch_indices[i] - self.num_samples + sample_ref
                if idx >= 0:
                    self.td_memory[idx] = td_errors[i]
        self.memory_lock.release()

    def expert_train(self, current_states, training_one_q, training_n_q, is_expert, margins,
                     expert_actions, training_weights, next_states, actions, rewards, dones,
                     sample_ref, norm_batch_indices, expert_batch_indices):
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
        cur_q, next_q, _ = self.global_agent.predict([current_states,
                                                      next_states,
                                                      EMPTY_IS_EXPERT,
                                                      EMPTY_MARGIN,
                                                      EMPTY_EXPERT_ACTIONS])
        self.training_episodes += 1
        if self.training_episodes % TARGET_UPDATE_FREQ == 0:
            self.update_target()
        if self.training_episodes % SAVE_FREQ == 0:
            self.save_model()
        self.global_lock.release()
        norm_td_errors = []
        expert_td_errors = []
        for i in range(BATCH_SIZE):
            a = actions[i]
            if dones[i]:
                td = (abs(rewards[i] - cur_q[i, a]) + PER_EPSILON) ** PER_ALPHA
            else:
                td = (abs(rewards[i] + DISCOUNT * np.max(next_q[i]) - cur_q[i, a]) + PER_EPSILON) ** PER_ALPHA
            if i < NORM_BATCH_SIZE:
                norm_td_errors.append(td)
            else:
                expert_td_errors.append(td)
        self.memory_lock.acquire()
        for i in range(NORM_BATCH_SIZE):
            if self.num_samples <= MAX_MEMORY:
                self.td_memory[norm_batch_indices[i]] = norm_td_errors[i]
            else:
                idx = norm_batch_indices[i] - self.num_samples + sample_ref
                if idx >= 0:
                    self.td_memory[idx] = norm_td_errors[i]
        self.memory_lock.release()
        self.expert_lock.acquire()
        for i in range(EXPERT_BATCH_SIZE):
            self.expert_td_memory[expert_batch_indices[i]] = expert_td_errors[i]
        self.expert_lock.release()

    def plot_score(self):
        score = np.array(self.score_memory.copy())
        num_games = len(score)
        games = list(range(num_games))
        ave_score = np.array(pd.DataFrame(score).ewm(span=num_games).mean()).squeeze()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(games, score, 'g', label='Score')
        ax.plot(games, ave_score, 'r', label='Average Score')
        ax.set_title('Score vs Number of Games Played')
        ax.set_ylabel('Score')
        ax.set_xlabel('Number of Games Played')
        ax.axis(xmin=0, xmax=num_games-1)
        ax.legend(loc='lower right', ncol=1)
        plt.savefig('Training_Plot_{}.png'.format(self.games_played))
        self.display_lock.acquire()
        print("________________________________________________________________")
        print('Data Plot Updated')
        self.display_lock.release()


class Player:
    def __init__(self, player_num):
        self.player_num = player_num
        self.name = 'Player {}'.format(player_num)
        if NUM_PLAYERS == 1:
            self.epsilon = MAX_EPSILON
        else:
            alpha = np.log(MIN_EPSILON) + player_num * (np.log(MAX_EPSILON) - np.log(MIN_EPSILON)) / (NUM_PLAYERS - 1)
            self.epsilon = np.exp(alpha)

    def choose_action(self, global_manager, state):
        q_values = global_manager.player_predict(self.player_num, np.expand_dims(state, axis=0))
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action = np.argmax(q_values)
        return action, q_values[action], np.max(q_values)

    def update_global_memory(self, global_manager, episode_batch, chosen_qs, max_qs, score):
        td_errors = []
        for i in range(len(episode_batch) - 1):
            q = episode_batch[i][2] + DISCOUNT * max_qs[i + 1] - chosen_qs[i]
            td = (abs(q) + PER_EPSILON) ** PER_ALPHA
            td_errors.append(td)
        td = (abs(episode_batch[-1][2] - chosen_qs[-1]) + PER_EPSILON) ** PER_ALPHA
        td_errors.append(td)
        global_manager.add_to_memory(episode_batch, td_errors, score, self.name, self.epsilon)

    def play_game(self, global_manager):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        global_manager.acquire_display_lock()
        print("{} has started playing!".format(self.name))
        global_manager.release_display_lock()
        while True:
            training_episodes = global_manager.get_training_episodes()
            if training_episodes > MAX_TRAINING_EPISODES:
                break
            episode_batch = []
            chosen_qs = []
            max_qs = []
            past_score = 0
            state = global_manager.env_reset(self.player_num)
            while True:
                action, chosen_q, max_q = self.choose_action(global_manager, state)
                next_state, info = global_manager.env_step(self.player_num, action)
                reward = get_reward(info, past_score)
                done, reward = get_done(info, reward)
                episode_batch.append([state, action, reward, done])
                chosen_qs.append(chosen_q)
                max_qs.append(max_q)
                if done:
                    self.update_global_memory(global_manager, episode_batch, chosen_qs, max_qs, 10 * info['score'])
                    global_manager.update_player_weights(self.player_num)
                    break
                state = next_state
                past_score = info['score']


class Learner:
    def __init__(self, learner_num):
        self.learner_num = learner_num
        self.name = "Learner {}".format(learner_num)

    def norm_learn(self, global_manager):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        global_manager.acquire_display_lock()
        print("{} has started learning!".format(self.name))
        global_manager.release_display_lock()
        while True:
            training_episodes = global_manager.get_training_episodes()
            if training_episodes > MAX_TRAINING_EPISODES:
                break
            global_manager.acquire_memory_lock()
            sample_ref = global_manager.get_num_samples()
            td_prob = global_manager.get_td_prob()
            batch_indices = np.random.choice(len(td_prob),
                                             BATCH_SIZE,
                                             replace=False,
                                             p=td_prob)
            current_states, actions, one_rewards, n_rewards, one_dones, n_dones, one_next_states, \
            n_next_states, exponents = global_manager.get_batch(batch_indices)
            global_manager.release_memory_lock()

            if USE_TRAINING_WEIGHTS:
                training_weights = []
                for idx in batch_indices:
                    training_weights.append(td_prob[idx] ** -PER_BETA)
                training_weights = np.array(training_weights) / np.max(training_weights)
            else:
                training_weights = np.ones(BATCH_SIZE)

            current_states = np.array(current_states)
            one_next_states = np.array(one_next_states)
            n_next_states = np.array(n_next_states)
            cur_q, one_next_q, n_next_q, one_target_q, n_target_q = global_manager.global_predict(current_states,
                                                                                                  one_next_states,
                                                                                                  n_next_states)
            training_one_q = cur_q.copy()
            training_n_q = cur_q.copy()
            for i in range(BATCH_SIZE):
                a = actions[i]
                if one_dones[i]:
                    training_one_q[i, a] = one_rewards[i]
                else:
                    max_a = np.argmax(one_next_q[i])
                    training_one_q[i, a] = one_rewards[i] + DISCOUNT * one_target_q[i, max_a]
                if n_dones[i]:
                    training_n_q[i, a] = n_rewards[i]
                else:
                    max_a = np.argmax(n_next_q[i])
                    training_n_q[i, a] = n_rewards[i] + (DISCOUNT ** exponents[i]) * n_target_q[i, max_a]
            global_manager.norm_train(current_states, training_one_q, training_n_q, training_weights,
                                      one_next_states, actions, one_rewards, one_dones, sample_ref, batch_indices)

    def expert_learn(self, global_manager):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        print("{} has started learning!".format(self.name))
        while True:
            training_episodes = global_manager.get_training_episodes()
            if training_episodes > MAX_TRAINING_EPISODES:
                break
            global_manager.acquire_memory_lock()
            sample_ref = global_manager.get_num_samples()
            norm_td_prob = global_manager.get_td_prob()
            norm_batch_indices = np.random.choice(len(norm_td_prob),
                                                  NORM_BATCH_SIZE,
                                                  replace=False,
                                                  p=norm_td_prob)
            norm_current_states, norm_actions, norm_one_rewards, \
            norm_n_rewards, norm_one_dones, norm_n_dones, norm_one_next_states, norm_n_next_states, \
            norm_exponents = global_manager.get_batch(norm_batch_indices)
            global_manager.release_memory_lock()
            global_manager.acquire_expert_lock()
            expert_td_prob = global_manager.get_expert_td_prob()
            expert_batch_indices = np.random.choice(len(expert_td_prob),
                                                    EXPERT_BATCH_SIZE,
                                                    replace=False,
                                                    p=expert_td_prob)
            expert_current_states, expert_actions, expert_one_rewards, \
            expert_n_rewards, expert_one_dones, expert_n_dones, expert_one_next_states, expert_n_next_states, \
            expert_exponents = global_manager.get_expert_batch(expert_batch_indices)
            global_manager.release_expert_lock()
            
            if USE_TRAINING_WEIGHTS:
                training_weights = []
                for n_idx in norm_batch_indices:
                    training_weights.append((NORM_FRACTION * norm_td_prob[n_idx]) ** -PER_BETA)
                for e_idx in expert_batch_indices:
                    training_weights.append((EXPERT_FRACTION * expert_td_prob[e_idx]) ** -PER_BETA)
                training_weights = np.array(training_weights) / np.max(training_weights)
            else:
                training_weights = np.ones(BATCH_SIZE)

            current_states = np.vstack((np.array(norm_current_states), np.array(expert_current_states)))
            one_next_states = np.vstack((np.array(norm_one_next_states), np.array(expert_one_next_states)))
            n_next_states = np.vstack((np.array(norm_n_next_states), np.array(expert_n_next_states)))
            actions = norm_actions + expert_actions
            one_rewards = norm_one_rewards + expert_one_rewards
            n_rewards = norm_n_rewards + expert_n_rewards
            one_dones = norm_one_dones + expert_one_dones
            n_dones = norm_n_dones + expert_n_dones
            exponents = norm_exponents + expert_exponents
            cur_q, one_next_q, n_next_q, one_target_q, n_target_q = global_manager.global_predict(current_states,
                                                                                                  one_next_states,
                                                                                                  n_next_states)
            training_one_q = cur_q.copy()
            training_n_q = cur_q.copy()
            is_expert = []
            margins = []
            expert_actions = []
            for i in range(BATCH_SIZE):
                a = actions[i]
                if i < NORM_BATCH_SIZE:
                    is_expert.append(0)
                else:
                    is_expert.append(1)
                margin = MARGIN * np.ones(NUM_ACTIONS)
                margin[a] = 0
                margins.append(margin)
                expert_action = np.zeros(NUM_ACTIONS)
                expert_action[a] = 1
                expert_actions.append(expert_action)
                if one_dones[i]:
                    training_one_q[i, a] = one_rewards[i]
                else:
                    max_a = np.argmax(one_next_q[i])
                    training_one_q[i, a] = one_rewards[i] + DISCOUNT * one_target_q[i, max_a]
                if n_dones[i]:
                    training_n_q[i, a] = n_rewards[i]
                else:
                    max_a = np.argmax(n_next_q[i])
                    training_n_q[i, a] = n_rewards[i] + (DISCOUNT ** exponents[i]) * n_target_q[i, max_a]
            global_manager.expert_train(current_states, training_one_q, training_n_q, np.array(is_expert),
                                        np.array(margins), np.array(expert_actions), training_weights,
                                        one_next_states, actions, one_rewards, one_dones, sample_ref,
                                        norm_batch_indices, expert_batch_indices)


if __name__ == '__main__':
    BaseManager.register('GlobalManager', GlobalManager)
    manager = BaseManager()
    manager.start()
    global_manager = manager.GlobalManager()
    players = []
    for i in range(NUM_PLAYERS):
        players.append(Player(i))
    player_processes = []
    for player in players:
        p = mp.Process(target=player.play_game, args=[global_manager])
        p.start()
        player_processes.append(p)
        time.sleep(0.1)
    global_manager.wait_for_warm_up()
    learners = []
    for i in range(NUM_LEARNERS):
        learners.append(Learner(i))
    learner_processes = []
    for learner in learners:
        if USE_EXPERT:
            l = mp.Process(target=learner.expert_learn, args=[global_manager])
            l.start()
        else:
            l = mp.Process(target=learner.norm_learn, args=[global_manager])
            l.start()
        learner_processes.append(l)
        time.sleep(0.1)
    for p in player_processes:
        p.join()
    for l in learner_processes:
        l.join()
import Neural_Networks as nn
import numpy as np
import cv2
import pandas as pd
import retrowrapper
import _pickle as pkl
import threading
import time
import zlib


# Debugging
RENDER = True
LOAD_WEIGHTS = True
PRINT_STATUS = True
USE_EXPERT_DATA = True
USE_TRAINING_WEIGHTS = True
COMPRESS_STATES = True


# File Paths
GAME_PATH = "/path/to/Game Files"
STATE_PATH = "/path/to/Game Files/Level1.state"
EXPERT_FOLDER = "Expert Data/"
EXPERT_STATES_PATH = EXPERT_FOLDER + "Expert_States.pkl"
COMP_STATES_PATH = EXPERT_FOLDER + "Compressed_Expert_States.pkl"
EXPERT_ACTIONS_PATH = EXPERT_FOLDER + "Expert_Actions.pkl"
EXPERT_REWARDS_PATH = EXPERT_FOLDER + "Expert_Rewards.pkl"
EXPERT_DONES_PATH = EXPERT_FOLDER + "Expert_Dones.pkl"
LOAD_WEIGHTS_PATH = "Pretrained_Global_Weights.h5"
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
L2_REG = 0  #5e-5
LEARNING_RATE = 1e-5
CLIP_NORM = 50.0
HUBER_DELTA = 1.0


# Players and Learners
NUM_PLAYERS = 2
MAX_EPSILON = 0.1
MIN_EPSILON = 2e-3
FRAME_SKIP = 4
FRAMES_PER_STEP = NUM_FRAMES * FRAME_SKIP
DISCOUNT = 0.993
REWARD_NORMALIZATION = 50.0


# Learners
NUM_LEARNERS = 1
MAX_MEMORY_SIZE = int(1e6)
MAX_TRAINING_EPISODES = int(1e5)
PER_EPSILON = 1e-5
PER_ALPHA = 0.4
PER_BETA = 0.6
BATCH_SIZE = 32
WARM_UP = 50 * BATCH_SIZE


# Expert Imitation Parameters
ROWS = np.arange(BATCH_SIZE).astype(int)
EXPERT_FRACTION = 0.25
EXPERT_BATCH_SIZE = int(EXPERT_FRACTION * BATCH_SIZE)
NORM_BATCH_SIZE = BATCH_SIZE - EXPERT_BATCH_SIZE


# Frequencies
UPDATE_TARGET_FREQUENCY = 2500
SAVE_FREQ = 1000
UPDATE_CSV_FREQ = 25


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
        self.global_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        self.learner_lock = threading.Lock()
        self.learner_lock.acquire()
        self.is_learner_locked = True
        if USE_EXPERT_DATA:
            self.expert_lock = threading.Lock()
        if PRINT_STATUS:
            self.display_lock = threading.Lock()

        # Initialize Neural Network
        self.global_agent = nn.create_global_model_v3(state_shape=STATE_SHAPE,
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
        self.global_target = nn.create_global_model_v3(state_shape=STATE_SHAPE,
                                                       num_actions=NUM_ACTIONS,
                                                       num_dense_units=NUM_DENSE_UNITS,
                                                       l2_regularizer=L2_REG,
                                                       learning_rate=LEARNING_RATE,
                                                       huber_delta=HUBER_DELTA,
                                                       clipnorm=CLIP_NORM)
        self.update_target()

        # Initialize Memory
        self.states = []
        self.actions = np.empty(0)
        self.rewards = np.empty(0)
        self.returns = np.empty(0)
        self.dones = np.empty(0)
        self.td = np.empty(0)
        if USE_EXPERT_DATA:
            if COMPRESS_STATES:
                with open(COMP_STATES_PATH, 'rb') as handle:
                    self.ex_states = pkl.load(handle)
            else:
                with open(EXPERT_STATES_PATH, 'rb') as handle:
                    self.ex_states = pkl.load(handle)
            with open(EXPERT_ACTIONS_PATH, 'rb') as handle:
                self.ex_actions = np.array(pkl.load(handle)).astype(int)
            with open(EXPERT_REWARDS_PATH, 'rb') as handle:
                self.ex_rewards = np.array(pkl.load(handle)) / 10.0
            with open(EXPERT_DONES_PATH, 'rb') as handle:
                self.ex_dones = np.array(pkl.load(handle))
            self.ex_returns = []
            for i in range(len(self.ex_rewards) - 1, -1, -1):
                if self.ex_dones[i]:
                    r = self.ex_rewards[i]
                else:
                    r = self.ex_rewards[i] + DISCOUNT * r
                self.ex_returns.append(r)
            self.ex_returns.reverse()
            self.ex_returns = np.array(self.ex_returns)
            self.ex_td = np.array(len(self.ex_actions) * [10])


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


class Player:
    def __init__(self, player_num):
        global global_agent
        # Counters
        self.player_num = player_num
        self.num_games_played = 0

        # How likely it is to explore
        if NUM_PLAYERS == 1:
            self.epsilon = MAX_EPSILON
        else:
            alpha = np.log(MIN_EPSILON) + player_num * (np.log(MAX_EPSILON) - np.log(MIN_EPSILON)) / (NUM_PLAYERS - 1)
            self.epsilon = np.exp(alpha)

        # Initialize Neural Network
        self.player = nn.create_player_model(state_shape=STATE_SHAPE,
                                             num_actions=NUM_ACTIONS,
                                             num_dense_units=NUM_DENSE_UNITS,
                                             l2_regularizer=L2_REG,
                                             learning_rate=LEARNING_RATE,
                                             clipnorm=CLIP_NORM,
                                             huber_delta=HUBER_DELTA)
        self.update_weights()

        # Initialize Game Environment
        self.env = retrowrapper.RetroWrapper(game=GAME_PATH,
                                             state=STATE_PATH)

    # Copies weights from global agent
    def update_weights(self):
        global global_agent
        global_agent.global_lock.acquire()
        self.player.set_weights(global_agent.global_agent.get_weights())
        global_agent.global_lock.release()

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
        global global_agent
        q_values = self.player.predict(np.expand_dims(state, axis=0)).squeeze()
        if np.random.random() < self.epsilon or global_agent.is_learner_locked:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action = np.argmax(q_values)
        return action, q_values[action], np.max(q_values)

    # Collects training data
    def play_game(self):
        global global_agent
        if PRINT_STATUS:
            global_agent.display_lock.acquire()
            print("Player {} has started playing!".format(self.player_num))
            global_agent.display_lock.release()
        while True:
            if global_agent.num_training_episodes > MAX_TRAINING_EPISODES:
                break
            states = []
            actions = np.empty(0)
            rewards = np.empty(0)
            dones = np.empty(0)
            chosen_qs = np.empty(0)
            max_qs = np.empty(0)
            past_score = 0
            state = self.env_reset()
            total_steps = 0
            total_reward = 0
            while True:
                action, chosen_q, max_q = self.choose_action(state)
                next_state, info = self.env_step(action)
                reward = get_reward(info, past_score)
                done, reward = get_done(info, reward)

                if COMPRESS_STATES:
                    state = zlib.compress(pkl.dumps(state))
                states.append(state)
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
            td = np.power(abs(q), PER_ALPHA) + PER_EPSILON
            returns = []
            r = 0
            for i in range(len(rewards) - 1, -1, -1):
                r = rewards[i] + DISCOUNT * r
                returns.append(r)
            returns.reverse()

            global_agent.memory_lock.acquire()
            global_agent.num_games_played += 1
            global_agent.num_samples_generated += len(td)

            global_agent.training_data.append([time.process_time() - global_agent.start_time,
                                               global_agent.num_games_played,
                                               global_agent.num_training_episodes,
                                               global_agent.num_samples_generated,
                                               total_reward,
                                               10 * info['score'],
                                               total_steps,
                                               self.epsilon])

            global_agent.states = global_agent.states + states
            global_agent.actions = np.append(global_agent.actions, actions)
            global_agent.rewards = np.append(global_agent.rewards, rewards)
            global_agent.returns = np.append(global_agent.returns, np.array(returns))
            global_agent.dones = np.append(global_agent.dones, dones)
            global_agent.td = np.append(global_agent.td, td)

            td_len = len(global_agent.td)
            if td_len > MAX_MEMORY_SIZE:
                diff = td_len - MAX_MEMORY_SIZE
                global_agent.states = global_agent.states[diff:]
                global_agent.actions = global_agent.actions[diff:]
                global_agent.rewards = global_agent.rewards[diff:]
                global_agent.returns = global_agent.returns[diff:]
                global_agent.dones = global_agent.dones[diff:]
                global_agent.td = global_agent.td[diff:]

            if global_agent.is_learner_locked and td_len > WARM_UP:
                global_agent.learner_lock.release()
                global_agent.is_learner_locked = False

            if len(global_agent.training_data) % UPDATE_CSV_FREQ == 0:
                global_agent.update_data_frame()

            global_agent.memory_lock.release()
            self.update_weights()


class Learner:
    def __init__(self, learner_num):
        self.learner_num = learner_num

    def learn(self):
        global global_agent
        if PRINT_STATUS:
            global_agent.display_lock.acquire()
            print("Learner {} has started learning!".format(self.learner_num))
            global_agent.display_lock.release()
        while True:
            if global_agent.num_training_episodes > MAX_TRAINING_EPISODES:
                break
            start_time = time.process_time()

            global_agent.memory_lock.acquire()
            len_memory = len(global_agent.td)
            sample_ref = global_agent.num_samples_generated
            td_prob = global_agent.td / np.sum(global_agent.td)
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
            next_indices = ((batch_indices + 1) % len_memory).copy()
            if COMPRESS_STATES:
                current_states = [pkl.loads(zlib.decompress(global_agent.states[k])) for k in batch_indices]
                next_states = [pkl.loads(zlib.decompress(global_agent.states[k])) for k in next_indices]
            else:
                current_states = [global_agent.states[k] for k in batch_indices]
                next_states = [global_agent.states[k] for k in next_indices]
            actions = global_agent.actions[batch_indices].astype(int)
            rewards = global_agent.rewards[batch_indices]
            returns = global_agent.returns[batch_indices]
            dones = global_agent.dones[batch_indices]
            global_agent.memory_lock.release()

            if USE_EXPERT_DATA:
                global_agent.expert_lock.acquire()
                num_ex_samples = len(global_agent.ex_td)
                ex_td_prob = global_agent.ex_td / np.sum(global_agent.ex_td)
                ex_batch_indices = np.random.choice(num_ex_samples,
                                                    EXPERT_BATCH_SIZE,
                                                    replace=False,
                                                    p=ex_td_prob)
                ex_next_indices = ((ex_batch_indices + 1) % num_ex_samples).copy()
                if COMPRESS_STATES:
                    ex_current_states=[pkl.loads(zlib.decompress(global_agent.ex_states[k])) for k in ex_batch_indices]
                    ex_next_states = [pkl.loads(zlib.decompress(global_agent.ex_states[k])) for k in ex_next_indices]
                else:
                    ex_current_states = [global_agent.ex_states[k] for k in ex_batch_indices]
                    ex_next_states = [global_agent.ex_states[k] for k in ex_next_indices]
                ex_actions = global_agent.ex_actions[ex_batch_indices]
                ex_rewards = global_agent.ex_rewards[ex_batch_indices]
                ex_returns = global_agent.ex_returns[ex_batch_indices]
                ex_dones = global_agent.ex_dones[ex_batch_indices]
                global_agent.expert_lock.release()

                current_states = current_states + ex_current_states
                next_states = next_states + ex_next_states
                actions = np.append(actions, ex_actions)
                rewards = np.append(rewards, ex_rewards)
                returns = np.append(returns, ex_returns)
                dones = np.append(dones, ex_dones)

            current_states = np.array(current_states)
            next_states = np.array(next_states)

            if USE_TRAINING_WEIGHTS:
                training_weights = td_prob[batch_indices]
                if USE_EXPERT_DATA:
                    training_weights = np.append((1 - EXPERT_FRACTION) * training_weights,
                                                 EXPERT_FRACTION * ex_td_prob[ex_batch_indices])
                training_weights = np.power(training_weights, -PER_BETA)
                training_weights = training_weights / np.max(training_weights)
            else:
                training_weights = np.ones(BATCH_SIZE)

            global_agent.global_lock.acquire()
            cur_q, next_q = global_agent.global_agent.predict([current_states,
                                                               next_states])
            target_q, _ = global_agent.global_target.predict([next_states,
                                                              next_states])
            global_agent.global_lock.release()

            training_one_q = cur_q.copy()
            training_n_q = cur_q.copy()

            max_a = np.argmax(next_q, axis=1)
            training_one_q[ROWS, actions] = rewards + DISCOUNT * (1 - dones) * target_q[ROWS, max_a]
            training_n_q[ROWS, actions] = returns

            global_agent.global_lock.acquire()
            global_agent.global_agent.fit([current_states,
                                           current_states],
                                          [training_one_q,
                                           training_n_q],
                                          batch_size=BATCH_SIZE,
                                          verbose=0,
                                          sample_weight=[training_weights,
                                                         training_weights])
            q, nq = global_agent.global_agent.predict([current_states,
                                                       next_states])
            global_agent.num_training_episodes += 1
            if global_agent.num_training_episodes % UPDATE_TARGET_FREQUENCY == 0:
                global_agent.update_target()
            if global_agent.num_training_episodes % SAVE_FREQ == 0:
                global_agent.save_model()
            global_agent.global_lock.release()

            chosen_q = q[ROWS, actions]
            r = rewards + DISCOUNT * (1 - dones) * np.max(nq, axis=1) - chosen_q
            td = np.power(abs(r), PER_ALPHA) + PER_EPSILON

            if USE_EXPERT_DATA:
                global_agent.expert_lock.acquire()
                global_agent.ex_td[ex_batch_indices] = td[NORM_BATCH_SIZE:]
                global_agent.expert_lock.release()

            global_agent.memory_lock.acquire()
            if USE_EXPERT_DATA:
                if global_agent.num_samples_generated > MAX_MEMORY_SIZE:
                    diff = global_agent.num_samples_generated - sample_ref
                    shifted_batch_indices = np.max(batch_indices - diff, -1)
                    global_agent.td[shifted_batch_indices] = np.where(shifted_batch_indices == -1,
                                                                      global_agent.td[shifted_batch_indices],
                                                                      td[:NORM_BATCH_SIZE])
                else:
                    global_agent.td[batch_indices] = td[:NORM_BATCH_SIZE]
            else:
                if global_agent.num_samples_generated > MAX_MEMORY_SIZE:
                    diff = global_agent.num_samples_generated - sample_ref
                    shifted_batch_indices = np.max(batch_indices - diff, -1)
                    global_agent.td[shifted_batch_indices] = np.where(shifted_batch_indices == -1,
                                                                      global_agent.td[shifted_batch_indices],
                                                                      td)
                else:
                    global_agent.td[batch_indices] = td
            global_agent.memory_lock.release()
            print(round(time.process_time() - start_time, 3))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    global_agent = Global_Agent()
    players = []
    learners = []
    for i in range(NUM_PLAYERS):
        players.append(Player(i))
    for j in range(NUM_LEARNERS):
        learners.append(Learner(j))
    player_threads = []
    for player in players:
        p = threading.Thread(target=player.play_game)
        p.start()
        player_threads.append(p)
        time.sleep(0.1)
    global_agent.learner_lock.acquire()
    learner_threads = []
    for learner in learners:
        l = threading.Thread(target=learner.learn)
        l.start()
        learner_threads.append(l)
        time.sleep(0.1)
    for thread in player_threads:
        thread.join()
    for thread in learner_threads:
        thread.join()
import Neural_Networks as nn
import numpy as np
import _pickle as pkl

GAME_PATH = "/path/to/Game Files"
STATE_PATH = "/path/to/Game Files/Level1.state"
EXPERT_FOLDER = "Expert Data/"
EXPERT_STATES_PATH = EXPERT_FOLDER + "Expert_States.pkl"
EXPERT_ACTIONS_PATH = EXPERT_FOLDER + "Expert_Actions.pkl"
EXPERT_REWARDS_PATH = EXPERT_FOLDER + "Expert_Rewards.pkl"
EXPERT_DONES_PATH = EXPERT_FOLDER + "Expert_Dones.pkl"

# Neural Network Parameters
WIDTH = 68
HEIGHT = 84
NUM_FRAMES = 2
STATE_SHAPE = (HEIGHT, WIDTH, NUM_FRAMES)
EXP_STATE_SHAPE = (1, HEIGHT, WIDTH, NUM_FRAMES)
NUM_ACTIONS = 4
NUM_DENSE_UNITS = 256
L2_REG = 5e-4
LEARNING_RATE = 1e-5
CLIP_NORM = 50.0
HUBER_DELTA = 1.0

# Training Parameters
BATCH_SIZE = 64
DISCOUNT = 0.993
IS_EXPERT = np.ones((BATCH_SIZE, 1))
EXPERT_ACTIONS = np.zeros((BATCH_SIZE, NUM_ACTIONS))
ROWS = np.arange(BATCH_SIZE).astype(int)
MARGIN = 1
EMPTY_LOSS = np.zeros((BATCH_SIZE, 1))


with open(EXPERT_STATES_PATH, 'rb') as handle:
    STATES = np.array(pkl.load(handle))
with open(EXPERT_ACTIONS_PATH, 'rb') as handle:
    ACTIONS = np.array(pkl.load(handle)).astype(int)
with open(EXPERT_REWARDS_PATH, 'rb') as handle:
    REWARDS = list(np.array(pkl.load(handle)) / 10.0)
with open(EXPERT_DONES_PATH, 'rb') as handle:
    DONES = list(pkl.load(handle))
NUM_SAMPLES = len(ACTIONS)
DONES.reverse()
REWARDS.reverse()

agent = nn.create_pretrain_model(state_shape=STATE_SHAPE,
                                 num_actions=NUM_ACTIONS,
                                 num_dense_units=NUM_DENSE_UNITS,
                                 l2_regularizer=L2_REG,
                                 learning_rate=LEARNING_RATE,
                                 huber_delta=HUBER_DELTA,
                                 clipnorm=CLIP_NORM)
returns = []
for i in range(NUM_SAMPLES):
    if DONES[i]:
        r = REWARDS[i]
    else:
        r = REWARDS[i] + DISCOUNT * r
    returns.append(r)
returns.reverse()
REWARDS.reverse()
DONES.reverse()
returns = np.array(returns)

for i in range(75000):
    indices = np.random.choice(NUM_SAMPLES,
                               BATCH_SIZE,
                               replace=False)
    states = STATES[indices]
    actions = ACTIONS[indices].astype(int)
    rewards = returns[indices]
    expert_actions = EXPERT_ACTIONS.copy()
    expert_actions[ROWS, actions] = 1
    margins = MARGIN * (1 - expert_actions)
    q, im_loss = agent.predict([states,
                                IS_EXPERT,
                                margins,
                                expert_actions])
    training_q = q.copy()
    training_q[ROWS, actions] = rewards

    agent.fit([states,
               IS_EXPERT,
               margins,
               expert_actions],
              [training_q,
               EMPTY_LOSS],
              batch_size=BATCH_SIZE,
              verbose=0)
    if i % 1000 == 0:
        agent.save_weights("Pretrained_Weights.h5")
        print(i)
        print(im_loss)
agent.save_weights("Pretrained_Weights.h5")
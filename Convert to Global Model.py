import Neural_Networks as nn


LOAD_WEIGHTS_PATH = "Pretrained_Weights.h5"
SAVE_GLOBAL_PATH = "Pretrained_Global_Weights.h5"
SAVE_PLAYER_PATH = "Pretrained_Player_Weights.h5"


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


pretrain_model = nn.create_pretrain_model(state_shape=STATE_SHAPE,
                                          num_actions=NUM_ACTIONS,
                                          num_dense_units=NUM_DENSE_UNITS,
                                          l2_regularizer=L2_REG,
                                          learning_rate=LEARNING_RATE,
                                          huber_delta=HUBER_DELTA,
                                          clipnorm=CLIP_NORM)
pretrain_model.load_weights(LOAD_WEIGHTS_PATH)

global_model = nn.create_global_model_v3(state_shape=STATE_SHAPE,
                                         num_actions=NUM_ACTIONS,
                                         num_dense_units=NUM_DENSE_UNITS,
                                         l2_regularizer=L2_REG,
                                         learning_rate=LEARNING_RATE,
                                         huber_delta=HUBER_DELTA,
                                         clipnorm=CLIP_NORM)
global_model.set_weights(pretrain_model.get_weights())
global_model.save_weights(SAVE_GLOBAL_PATH)

player_model = nn.create_player_model(state_shape=STATE_SHAPE,
                                      num_actions=NUM_ACTIONS,
                                      num_dense_units=NUM_DENSE_UNITS,
                                      l2_regularizer=L2_REG,
                                      learning_rate=LEARNING_RATE,
                                      clipnorm=CLIP_NORM,
                                      huber_delta=HUBER_DELTA)
player_model.set_weights(pretrain_model.get_weights())
player_model.save_weights(SAVE_PLAYER_PATH)
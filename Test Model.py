import retro
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K


GAME_PATH = "Game Files"
STATE_PATH = "Game Files/Level1.state"
LOAD_WEIGHTS_PATH = "Expert_Player_Weights_2.h5"

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


def format_state(state):
    state = state[14:218, 6:162, :]
    state = cv2.resize(state, (WIDTH, HEIGHT), cv2.INTER_CUBIC)
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    return state / 255


def get_done(info):
    if info['level'] > 0 or info['lives'] < 3:
        return True
    else:
        return False


def dueling_output(input_layer):
    state_layer = input_layer[0]
    action_layer = input_layer[1]
    return state_layer + action_layer - K.mean(action_layer, axis=1, keepdims=True)


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
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1,
                                                     clipnorm=1),
                  loss='mse')
    model.make_predict_function()
    return model


class Player:
    def __init__(self):
        self.env = retro.make(game=GAME_PATH, state=STATE_PATH)
        self.agent = create_player_model()
        self.agent.load_weights(LOAD_WEIGHTS_PATH)
        print(self.agent.summary())

    def env_reset(self):
        full_state = np.zeros(STATE_SHAPE)
        state = self.env.reset()
        self.env.render()
        full_state[:, :, 0] = format_state(state)
        n = 1
        for i in range(1, NUM_FRAMES * FRAME_SKIP):
            state, _, _, info = self.env.step(NO_ACTION)
            if i % FRAME_SKIP == 0:
                full_state[:, :, n] = format_state(state)
                n += 1
            self.env.render()
        return full_state

    def env_step(self, action):
        full_state = np.zeros(STATE_SHAPE)
        n = 0
        for i in range(NUM_FRAMES * FRAME_SKIP):
            state, _, _, info = self.env.step(ACTION_SET[action])
            if i % FRAME_SKIP == 0:
                full_state[:, :, n] = format_state(state)
                n += 1
            self.env.render()
        return full_state, info

    def choose_action(self, state):
        if np.random.random() < 0:
            return np.random.randint(NUM_ACTIONS)
        else:
            q_values = self.agent.predict(np.expand_dims(state, axis=0)).squeeze()
            print(q_values)
            return np.argmax(q_values)

    def play_game(self):
        state = self.env_reset()
        self.env.render()
        while True:
            action = self.choose_action(state)
            state, info = self.env_step(action)
            done = get_done(info)
            #done = False
            if done:
                break
        print('Done!')


player = Player()
player.play_game()
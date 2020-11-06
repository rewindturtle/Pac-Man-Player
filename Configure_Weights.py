import tensorflow as tf
import tensorflow.keras.backend as K


"""
Converts the larger neural network used in training to a smaller neural network that only has
one output
"""


# Neural Network Parameters
WIDTH = 68
HEIGHT = 84
NUM_FRAMES = 2
FRAME_SKIP = 4
STATE_SHAPE = (HEIGHT, WIDTH, NUM_FRAMES)
NUM_ACTIONS = 4
DENSE_UNITS = 512
L2_REG = 5e-5

LOAD_WEIGHTS_PATH = "Expert_Weights.h5"


def dueling_output(input_layer):
    state_layer = input_layer[0]
    action_layer = input_layer[1]
    return state_layer + action_layer - K.mean(action_layer, axis=1, keepdims=True)


def expert_output(margin_input):
    is_expert = margin_input[0]  # 1 if expert action, 0 otherwise
    expert_margin = margin_input[1]  # Defines the l_loss
    expert_action = margin_input[2]  # one-hot vector of the action take by the expert
    pred_q_values = margin_input[3]  # the q values predicted by the current nn
    expert_q = K.sum(pred_q_values * expert_action, axis=1)
    max_margin = K.max(pred_q_values + expert_margin, axis=1)
    max_loss = max_margin - expert_q
    r_max_loss = K.reshape(max_loss, K.shape(is_expert))
    return is_expert * r_max_loss


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
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1,
                                                     clipnorm=1),
                  loss=['mse', 'mse', 'mae'])
    model.make_predict_function()
    model.make_train_function()
    return model


model_1 = create_global_model()
model_2 = create_player_model()
model_1.load_weights(LOAD_WEIGHTS_PATH)
model_2.set_weights(model_1.get_weights())
model_2.save_weights('Player_Weights.h5')
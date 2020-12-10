import tensorflow as tf


# The output of the neural network that calculates the q-values based on the dueling network architecture
# Q(s, a) = V(s) + A(s,a) - mean_a(A(s,a))
def dueling_output(input_layer):
    state_layer = input_layer[0]
    action_layer = input_layer[1]
    return state_layer + action_layer - tf.keras.backend.mean(action_layer, axis=1, keepdims=True)


# Output that compares the argmax Q value action predicted by the nn and the action taken by the expert
def expert_output(margin_input):
    is_expert = margin_input[0]  # 1 if expert action, 0 otherwise
    expert_margin = margin_input[1]  # Defines the l_loss
    expert_action = margin_input[2]  # one-hot vector of the action take by the expert
    pred_q_values = margin_input[3]  # the q values predicted by the current nn
    expert_q = tf.keras.backend.sum(pred_q_values * expert_action, axis=1, keepdims=True)
    relu_values = tf.keras.backend.relu(pred_q_values + expert_margin - expert_q)
    mean_values = tf.keras.backend.mean(relu_values, axis=1, keepdims=True)
    return is_expert * mean_values


# The base neural network that the global, player, and learner networks are built from.
def create_base_model(state_shape, num_actions, num_dense_units, l2_regularizer):
    input_layer = tf.keras.layers.Input(shape=state_shape,
                                        name="Input_Layer")
    conv_1 = tf.keras.layers.Conv2D(16,
                                    kernel_size=(8, 8),
                                    strides=4,
                                    kernel_initializer='he_uniform',
                                    input_shape=state_shape,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                    bias_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                    name='Conv_Layer_1')(input_layer)
    conv_2 = tf.keras.layers.Conv2D(32,
                                    kernel_size=(4, 4),
                                    strides=2,
                                    kernel_initializer='he_uniform',
                                    input_shape=state_shape,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                    bias_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                    name='Conv_Layer_2')(conv_1)
    conv_3 = tf.keras.layers.Conv2D(32,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    kernel_initializer='he_uniform',
                                    input_shape=state_shape,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                    bias_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                    name='Conv_Layer_3')(conv_2)
    flat = tf.keras.layers.Flatten(name='Flat_Layer')(conv_3)
    state_dense = tf.keras.layers.Dense(num_dense_units,
                                        activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                        bias_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                        kernel_initializer='he_uniform',
                                        name='State_Dense_Layer')(flat)
    state_value = tf.keras.layers.Dense(1,
                                        kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                        bias_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                        name='State_Value_Layer')(state_dense)
    action_dense = tf.keras.layers.Dense(num_dense_units,
                                         activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                         bias_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                         kernel_initializer='he_uniform',
                                         name='Action_Dense_Layer')(flat)
    action_value = tf.keras.layers.Dense(num_actions,
                                         activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                         bias_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                         kernel_initializer='he_uniform',
                                         name='Action_Value_Layer')(action_dense)
    output_layer = tf.keras.layers.Lambda(dueling_output,
                                          name='Output_Layer')([state_value, action_value])
    base_model = tf.keras.Model(input_layer,
                                output_layer,
                                name='Base_Model')
    return base_model


def create_global_model(state_shape, num_actions, num_dense_units, l2_regularizer,
                        learning_rate, huber_delta, clipnorm):
    base_model = create_base_model(state_shape=state_shape, num_actions=num_actions,
                                   num_dense_units=num_dense_units, l2_regularizer=l2_regularizer)
    one_step_input = tf.keras.layers.Input(shape=state_shape,
                                           name='One_Step_Input_Layer')
    n_step_input = tf.keras.layers.Input(shape=state_shape,
                                         name='N_Step_Input_Layer')
    is_expert_input = tf.keras.layers.Input(shape=(1,),
                                            name='Is_Expert_Input')
    margin_input = tf.keras.layers.Input(shape=(num_actions,),
                                         name='Margin_Input')
    expert_action_input = tf.keras.layers.Input(shape=(num_actions,),
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
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate,
                                                     clipnorm=clipnorm),
                  loss=[tf.keras.losses.Huber(delta=huber_delta),
                        tf.keras.losses.Huber(delta=huber_delta),
                        'mae'])
    return model


# Model that makes decisions while playing the game
def create_player_model(state_shape, num_actions, num_dense_units, l2_regularizer, learning_rate,
                        huber_delta, clipnorm):
    model = create_base_model(state_shape, num_actions, num_dense_units, l2_regularizer)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate,
                                                     clipnorm=clipnorm),
                  loss=tf.keras.losses.Huber(delta=huber_delta))
    return model
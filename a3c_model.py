import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model


def build_policy_and_value_networks(num_actions, input_shape):
    state = tf.placeholder("float", [None, input_shape])

    inputs = Input(shape=([input_shape]))

    shared = Dense(name="Layer1", units=64, activation='relu')(inputs)

    action_probs = Dense(name="p", units=num_actions,
                         activation='softmax')(shared)
    state_value = Dense(name="v", units=1,
                        activation='linear')(shared)

    policy_network = Model(input=inputs, output=action_probs)
    value_network = Model(input=inputs, output=state_value)

    p_params = policy_network.trainable_weights
    v_params = value_network.trainable_weights

    p_out = policy_network(state)
    v_out = value_network(state)

    return state, p_out, v_out, p_params, v_params

import numpy as np
import viterbi
import data
from tqdm import tqdm
from logger import Logger
import datetime
import tensorflow as tf
import argparse

EPS = 1e-6


def keras_model(action_dim, z_dim):

    from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, Dot, Reshape, Softmax
    from beta_regularizer import BetaRegularization
    s = Input(shape=(110, 84, 1), name='input_s')
    z = Input(shape=(z_dim,), name='input_z')
    conv1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(s)
    conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv1)
    conv2_flat = Flatten()(conv2)
    h1 = Dense(256, activation='relu')(conv2_flat)

    # Predict the next latent
    h2_z = Dense(z_dim * z_dim, activation=None)(h1)  # (B x Z * Z)
    h2_z_reshaped = Reshape((z_dim, z_dim))(h2_z)  # (B x Z x Z)
    z_tp1_matrix = Softmax(axis=-1, name='latent_matrix')(h2_z_reshaped)
    z_tp1 = Dot(axes=1, name='latent')([z_tp1_matrix, z])

    # Predict the next action
    h2_a = Dense(action_dim * z_dim, activation=None)(h1)  # (B x A * Z)
    h2_a_reshaped = Reshape((z_dim, action_dim))(h2_a)       # (B x Z x A)
    a_matrix = Softmax(axis=-1, name='action_matrix')(h2_a_reshaped)
    a = Dot(axes=1, name='action')([a_matrix, z])

    # Predict termination
    h2_b = Dense(z_dim * 2, activation=None)(h1)  # (B x 2 * Z)
    h2_b_reshaped = Reshape((z_dim, 2))(h2_b)     # (B x Z x 2)
    b_matrix = Softmax(axis=-1, name='termination_matrix')(h2_b_reshaped)
    b_matrix = BetaRegularization(1.0, 99.)(b_matrix)
    b = Dot(axes=1, name='termination')([b_matrix, z])

    return tf.keras.Model(inputs=[s, z], outputs=[a, z_tp1, b])


def m_step(game, iteration, logger, transfer_weights, batch_size, z_dim):
    print('M step: %s (%d)' % (game, iteration))
    action_dim = len(data.get_gym_actions(game))

    ### Setup the session
    tf.keras.backend.clear_session()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2  # TODO: investigate dynamic memory allocation (allow growth = True)
    tf.keras.backend.set_session(tf.Session(config=config))

    # Load Dataset
    training_set, train_size = data.tfdata_generator(game, is_training=True, iteration=iteration, z_dim=z_dim, batch_size=batch_size, logger=logger)
    testing_set, test_size = data.tfdata_generator(game, is_training=False, iteration=iteration, z_dim=z_dim, batch_size=batch_size, logger=logger)
    print('Training size: %d' % train_size)
    print('Testing size: %d' % test_size)

    # Train Model
    model = keras_model(action_dim, z_dim)
    if transfer_weights and iteration > 1:
        logger.load_weights(iteration - 1, model)

    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
                  metrics=['acc'])
    [s_train, z_train, a_train, z_tp1_train, b_train] = training_set.make_one_shot_iterator().get_next()
    [s_test, z_test, a_test, z_tp1_test, b_test] = testing_set.make_one_shot_iterator().get_next()
    history = model.fit(
        [s_train, z_train],
        [a_train, z_tp1_train, b_train],
        steps_per_epoch=train_size // batch_size,
        epochs=1,
        validation_data=([s_test, z_test], [a_test, z_tp1_test, b_test]),
        validation_steps=test_size // batch_size,
        verbose=1)


    for (key, value) in history.history.items():
        assert len(value) == 1
        logger.log(iteration, key, value[0])
    logger.save_weights(iteration, model)

def _check_prob_matrix(P):
    '''Verifies that P is a valid Markov matrix, with last dimension summing to 1.'''
    assert len(P.shape) == 3
    assert np.allclose(np.sum(P, axis=2), 1)
    assert np.all(0 - EPS <= P) and np.all(P <= 1 + EPS)


def e_step(game, iteration, logger, batch_size, z_dim, learn_termination):
    '''iteration is the last model iteration. We'll save the latents as iteration + 1'''
    print('E step: %s (%d)' % (game, iteration))
    model = logger.load_model(iteration)
    predict_model = tf.keras.Model(inputs=[model.input[0]],
                                   outputs=[model.get_layer('action_matrix').output,
                                            model.get_layer('latent_matrix').output,
                                            model.get_layer('termination_matrix').output],)

    metrics = {}
    for traj_index in tqdm(data.get_traj_index_vec(game)):
        image_vec = data.load_images(game, traj_index)
        size = len(image_vec)

        image_vec = tf.constant(image_vec)
        dataset = tf.data.Dataset.from_tensor_slices(image_vec)
        # Transform and batch data at the same time
        dataset = dataset.map(data.image_preprocess_fn, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        s = dataset.make_one_shot_iterator().get_next()
        steps = int(np.ceil(float(size) / float(batch_size)))
        [action_probs, latent_probs, termination_probs] = predict_model.predict([s], steps=steps)
        assert len(action_probs) == size
        _check_prob_matrix(action_probs)
        _check_prob_matrix(latent_probs)
        _check_prob_matrix(termination_probs)

        # Compute the node potentials
        a_vec = np.array(data.load_actions(game, traj_index)).flatten()
        action_probs = action_probs[range(size), :, a_vec]  # T x Z
        # Note: We log-scale the node potentials, because we use a sum-based
        # version of Viterbi.
        node_potentials = np.log(action_probs)

        # Compute the edge potentials
        p_h_terminate = latent_probs[:-1]
        p_h_continue = np.eye(z_dim)[None, :, :]
        # termination_probs[..., 1] is probability that we *do* terminate
        if learn_termination:
            p_h = termination_probs[:-1, :, 1, None] * p_h_terminate + termination_probs[:-1, :, 0, None] * p_h_continue
        else:
            p_h = p_h_terminate
        _check_prob_matrix(p_h)

        edge_potentials = np.log(p_h)
        (z_vec, objective) = viterbi.viterbi(edge_potentials, node_potentials)

        # For the selected skills, what is the probability of the true action?
        action_probs_pos_mean = np.mean(action_probs[range(size), z_vec])
        # For the non-selected skills, what is the probability of the true action?
        action_probs_neg_mean = (np.sum(action_probs)  - size * action_probs_pos_mean) / (size * (z_dim - 1.0))

        num_empty = z_dim - len(set(z_vec))
        latent_switches = np.where(z_vec[:-1] != z_vec[1:])[0]
        avg_length = np.mean(latent_switches[1:] - latent_switches[:-1])
        avg_num_actions = np.mean([len(set(a_vec[z_vec == z])) for z in range(z_dim)])

        for (key, value) in [('empty_skills', num_empty),
                             ( 'skill_duration', avg_length),
                             ( 'actions_per_skill', avg_num_actions),
                             ( 'viterbi_objective', objective),
                             ( 'action_prob_pos_mean', action_probs_pos_mean),
                             ( 'action_prob_neg_mean', action_probs_neg_mean)]:
            metrics[key] = metrics.get(key, []) + [value]

        logger.save_z(iteration, traj_index, z_vec)

    print('Assignment metrics (%d):' % iteration)
    for (key, value_vec) in metrics.items():
        logger.log(iteration, key, np.mean(value_vec))
        print('\t%s = %.2f' % (key, np.mean(value_vec)))

def get_args():
    parser = argparse.ArgumentParser(description='Option Discovery')
    parser.add_argument('-game', type=str, default='revenge')
    parser.add_argument('-alpha', type=float, default=1.0)
    parser.add_argument('-beta', type=float, default=1.0)
    parser.add_argument('-z_dim', type=int, default=4)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-transfer_weights', type=bool, default=True)
    parser.add_argument('--no_learn_termination', dest='learn_termination', action='store_false')
    parser.set_defaults(learn_termination=True)
    return parser.parse_args()
 

if __name__ == '__main__':
    args = get_args()
    log_folder = 'logs/%s' % datetime.datetime.now().isoformat().replace(':', '-')
    logger = Logger(log_folder, vars(args))
    data.initialize_latents(args.game, logger, args.z_dim)
    for iteration in range(1, 101):
        m_step(args.game, iteration, logger, args.transfer_weights, args.batch_size, args.z_dim)
        e_step(args.game, iteration, logger, args.batch_size, args.z_dim, args.learn_termination)
        logger.save()

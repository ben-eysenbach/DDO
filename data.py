import tensorflow as tf
import numpy as np
import gym
import os
import glob
from tqdm import tqdm
import functools



AGC_ACTIONS = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

@functools.lru_cache()
def get_gym_actions(game):
    if game == 'revenge':
        name = 'MontezumaRevengeNoFrameskip-v4'
    elif game == 'mspacman':
        name = 'MsPacmanNoFrameskip-v4'
    elif game == 'pinball':
        name = 'VideoPinballNoFrameskip-v4'
    elif game == 'qbert':
        name = 'QbertNoFrameskip-v4'
    elif game == 'spaceinvaders':
        name = 'SpaceInvadersNoFrameskip-v4'
    else:
        raise ValueError('Unknown game: %s' % game)
    env = gym.make(name)
    gym_actions = env.env.get_action_meanings()
    return gym_actions


def image_preprocess_fn(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image = tf.image.rgb_to_grayscale(image_decoded)
    image = tf.image.resize_images(image, [110, 84])
    image = tf.cast(image, tf.float32) * (1. / 255)
    return image


def get_traj_index_vec(game):
    traj_index_vec = os.listdir(os.path.join('atari_v2_release/screens', game))
    traj_index_vec = [int(traj_index) for traj_index in traj_index_vec]
    traj_index_vec = sorted(traj_index_vec)
    # TODO HACK (truncated for fast debugging)
    return traj_index_vec[:100]


def load_actions(game, traj_index):
    filename = 'atari_v2_release/trajectories/%s/%d.txt' % (game, traj_index)
    action_fn = get_action_fn(game)
    a_vec = []
    with open(filename) as f:
        lines_vec = f.readlines()
        for line in lines_vec[2:]:
            items = line.split(',')
            agc_a = int(items[-1])
            gym_a = action_fn(agc_a)
            a_vec.append(gym_a)
    return a_vec


def load_images(game, traj_index):
    image_vec = glob.glob('atari_v2_release/screens/%s/%d/*.png' % (game, traj_index))
    def key(filename):
        return int(os.path.basename(filename).split('.')[0])
        
    image_vec = sorted(image_vec, key=key)
    return image_vec



def get_action_fn(game):
    gym_actions = get_gym_actions(game)
    def convert_fn(agc_a):
        token = AGC_ACTIONS[agc_a]
        if token in gym_actions:
            gym_a = gym_actions.index(token)
        else:
            gym_a = gym_actions.index('NOOP')
        return [gym_a]
    return convert_fn


def load_data(game, is_training, iteration, logger):
    '''We do a 80/20 train/test split.
        train - traj_index % 5 = 1, 2, 3, 4
        test  - traj_index % 5 = 0
    '''
    print('\tLoading data: (%s, %s, %d)' % (game, 'train' if is_training else 'val', iteration))
    image_vec_combined = []
    z_vec_combined = []
    a_vec_combined = []
    for traj_index in tqdm(get_traj_index_vec(game)):
        if (is_training and traj_index % 5 != 0) or (not is_training and traj_index % 5 == 0):
            a_vec = load_actions(game, traj_index)
            z_vec = logger.load_z(iteration - 1, traj_index)
            image_vec = load_images(game, traj_index)
            assert len(a_vec) == len(z_vec), 'Should have same number of actions and latents (%d vs %d)' % (len(a_vec), len(z_vec))
            assert len(a_vec) == len(image_vec), 'Should have same number of actions and images (%d vs %d)' % (len(a_vec), len(image_vec))
            a_vec_combined.extend(a_vec)
            z_vec_combined.extend(z_vec)
            image_vec_combined.extend(image_vec)
    assert len(image_vec_combined) == len(a_vec_combined)
    assert len(image_vec_combined) == len(z_vec_combined)
    return (image_vec_combined, z_vec_combined, a_vec_combined)


def tfdata_generator(game, is_training, iteration, z_dim, batch_size, logger):
    '''Construct a data generator using tf.Dataset'''

    (image_vec, z_vec, a_vec) = load_data(game, is_training, iteration, logger)
    # Compute the next step latent
    z_tp1_vec = np.hstack([z_vec[1:], [z_vec[-1]]])  # Assume that the latent after termination is same as last latent
    z_tp1_vec = z_tp1_vec[:, None]  # Add an extra dimension because TF expects labels to be 2D
    assert len(z_tp1_vec) == len(z_vec)

    # Compute the termination indicators
    z_vec = np.array(z_vec)
    b_vec = (z_vec[:-1] != z_vec[1:]).astype(int)
    b_vec = np.hstack([b_vec, [0]])  # Assume that the last transition is not terminating
    b_vec = b_vec[:, None]  # Add an extra dimension because TF expects labels to be 2D
    assert len(image_vec) == len(b_vec)

    size = len(image_vec)
    image_vec = tf.constant(image_vec)
    z_vec = tf.constant(z_vec)
    a_vec = tf.constant(a_vec)
    z_tp1_vec = tf.constant(z_tp1_vec)
    b_vec = tf.constant(b_vec)
    dataset = tf.data.Dataset.from_tensor_slices((image_vec, z_vec, a_vec, z_tp1_vec, b_vec))
    if is_training:
        dataset = dataset.shuffle(1000)  # depends on sample size


    def preprocess_fn(filename, latent, action, latent_tp1, termination):
        # Note: We don't encode the next latent
        image = image_preprocess_fn(filename)
        latent = tf.one_hot(latent, z_dim)
        return image, latent, action, latent_tp1, termination

    # Transform and batch data at the same time
    dataset = dataset.map(preprocess_fn, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)  # Note, batch size may be smaller for last batch

    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset, size


def initialize_latents(game, logger, z_dim):
    '''Latent #5 were generated with the model at iteration #4'''
    print('Initializing Latents')
    for traj_index in tqdm(get_traj_index_vec(game)):
        a_vec = load_actions(game, traj_index)
        size = len(a_vec)
        z_vec = np.random.choice(z_dim, size)
        logger.save_z(0, traj_index, z_vec)



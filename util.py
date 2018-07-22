import os, time

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def get_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config

def load_ckpt(sess, saver, ckpt_dir='train'):

    '''Load checkpoint from the ckpt_dir and restore it to saver and sess, waiting 10 secs in the case of failure.'''

    while True:
        try:
            ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            latest_filename = 'best.ckpt' if ckpt_dir == 'eval' else None
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            saver.restore(sess, ckpt_state.model_checkpoint_path)

            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)

            return ckpt_state.model_checkpoint_path
        except:
            tf.logging.info('Failed to load checkpoint from %s. Sleeping for %i secs...', ckpt_dir, 10)

            time.sleep(10)

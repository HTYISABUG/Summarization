import os

import tensorflow as tf

from data import Vocab

FLAGS = tf.app.flags.FLAGS

# Hyperparams
tf.app.flags.DEFINE_string('mode',       'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_string('data_path',  '',      'Path to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '',      'Path to vocabulary file.')
tf.app.flags.DEFINE_string('log_root',   '',      'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name',   '',      'Name for experiment. Logs will be saved in this directory under log_root.')

tf.app.flags.DEFINE_integer('hidden_dim',    256,   'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('batch_size',    16,    'batch size')
tf.app.flags.DEFINE_integer('emb_dim',       300,   'dimension of word embeddings')
tf.app.flags.DEFINE_integer('max_enc_steps', 400,   'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100,   'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('vocab_size',    50000, 'size of vocabulary')

tf.app.flags.DEFINE_float('lr',                 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc',   0.1,  'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform initializer')
tf.app.flags.DEFINE_float('trun_norm_init_std', 1e-4, 'std of truncated normal initializer')
tf.app.flags.DEFINE_float('max_grad_norm',      2.0,  'for gradient clipping')

def main(unused_args):
    if len(unused_args) != 1: raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.info('Starting pointer generator in %s mode...', (FLAGS.mode))

    # setup log directory
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)

    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == 'train': os.makedirs(FLAGS.log_root)
        else: raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    # setup vocabulary
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)

    # setup hps
    hps_name = ['mode',
                'hidden_dim', 'batch_size', 'emb_dim', 'max_enc_steps', 'max_dec_steps',
                'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trun_norm_init_std', 'max_grad_norm']
    hps = {}

    for k, v in iter(FLAGS.__flags):
        if k in hps_name: hps[k] = v

if __name__ == "__main__":
    tf.app.run()

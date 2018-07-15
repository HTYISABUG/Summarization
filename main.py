import os, time
from pprint import pprint
from collections import namedtuple

import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tfdbg

from data import Vocab
from batcher import Batcher
from model import Model

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

tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval dir and save it in the train dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

def run_training(model, batcher, emb):

    '''Setup and repeatedly runs training iterations, logging loss to screen and writing summaries'''

    train_dir = os.path.join(FLAGS.log_root, 'train')
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build()

    saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=train_dir, saver=saver, global_step=model.global_step)

    tf.logging.info('Preparing or waiting for session...')

    sess_context_manager = sv.prepare_or_wait_for_session(config=get_config())

    tf.logging.info('Created session.')

    try:
        with sess_context_manager as sess:
            if FLAGS.debug:
                sess = tfdbg.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter('has_inf_or_nan', tfdbg.has_inf_or_nan)

            model.load_embs(sess, emb)

            tf.logging.info('starting run_training')

            while True:
                batch = batcher.next_batch()

                tf.logging.info('running training step...')

                ts = time.time()
                result = model.run_train_step(sess, batch)

                tf.logging.info('seconds for training step: %.3f', time.time() - ts)

                tf.logging.info('loss: %f', result['loss'])

                if not np.isfinite(result['loss']):
                    raise Exception('Loss is not finite. Stopping.')

                if result['global_step'] % 100 == 0:
                    sv.summary_writer.flush()

    except KeyboardInterrupt as e:
        tf.logging.info('Caught keyboard interrupt on worker. Stopping supervisor...')
        sv.stop()

def get_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config

def run_eval(model, batcher):

    '''Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far.'''

    eval_dir = os.path.join(FLAGS.log_root, 'eval')
    if not os.path.exists(train_dir): os.makedirs(eval_dir)
    bestmodel_path = os.path.join(eval_dir, 'bestmodel')

    model.build()

    sess = tf.Session(config=get_config())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(eval_dir)

    running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None  # will hold the best loss achieved so far

    while True:
        load_ckpt(sess, saver)

        batch = batcher.next_batch()

        ts = time.time()
        result = model.run_eval_step(sess, batch)
        tf.logging.info('seconds for batch: %.2f', time.time() - ts)

        tf.logging.info('loss: %f', result['loss'])

        summary = result['summary']
        global_step = result['global_step']
        writer.add_summary(summary, global_step)

        running_avg_loss = calc_running_avg_loss(running_avg_loss, np.asscalar(loss), writer, global_step)

        if best_loss is None or running_avg_loss < bestmodel_path:
            tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_path)

            saver.save(sess, bestmodel_path, global_step=global_step, latest_filename='best.ckpt')
            best_loss = running_avg_loss

        if global_step % 100 == 0:
            writer.flush()

def load_ckpt(sess, saver, ckpt_dir='train'):

    '''Load checkpoint from the ckpt_dir and restore it to saver and sess, waiting 10 secs in the case of failure.'''

    while True:
        try:
            ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            latest_filename = 'best.ckpt' if ckpt_dir == 'eval' else None
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            saver.restore(sess, ckpt_state.model_checkpoint_path)

            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        except:
            tf.logging.info('Failed to load checkpoint from %s. Sleeping for %i secs...', ckpt_dir, 10)

            time.sleep(10)

def calc_running_avg_loss(running_avg_loss, loss, writer, step, decay=0.99):

    '''Calculate the running average loss via exponential decay.
       This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
        running_avg_loss: running_avg_loss so far
        loss: loss on the most recent eval step
        writer: FileWriter object to write for tensorboard
        step: training iteration step
        decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
        running_avg_loss: new running average loss
    '''

    running_avg_loss = loss if running_avg_loss == 0 else running_avg_loss * decay + loss * (1 - decay)
    running_avg_loss = min(running_avg_loss, 12) # clip

    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value(tag=tag_name, simple_value=running_avg_loss)
    writer.add_summary(loss_sum, step)

    tf.logging.info('running_avg_loss: %f', running_avg_loss)

    return running_avg_loss

def main(unused_args):
    if len(unused_args) != 1: raise Exception('Problem with flags: %s' % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting pointer generator in %s mode...', (FLAGS.mode))

    # setup log directory
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)

    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == 'train': os.makedirs(FLAGS.log_root)
        else: raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    # setup vocabulary
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size, FLAGS.emb_dim)

    # setup hps
    hps_name = ['mode',
                'hidden_dim', 'batch_size', 'emb_dim', 'max_enc_steps', 'max_dec_steps', 'vocab_size',
                'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trun_norm_init_std', 'max_grad_norm']
    hps = {}

    for k, v in FLAGS.__flags.items():
        if k in hps_name: hps[k] = v.value

    hps = namedtuple('HyperParams', hps.keys())(**hps)

    batcher = Batcher(FLAGS.data_path, vocab, hps, FLAGS.single_pass)

    tf.set_random_seed(13131)

    if FLAGS.mode == 'train':
        model = Model(hps)
        run_training(model, batcher, vocab.embedding)
    elif FLAGS.mode == 'eval':
        model = Model(hps)
        run_eval(model, batcher)
    elif FLAGS.mode == 'decode':
        hps['max_dec_steps'] = 1
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

if __name__ == '__main__':
    tf.app.run()

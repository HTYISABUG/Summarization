import os, time, sys
from pprint import pprint
from collections import namedtuple

import tensorflow as tf
import numpy as np

import util
from data import Vocab
from batcher import Batcher
from model import Model
from beam_search_decoder import BeamSearchDecoder

FLAGS = tf.app.flags.FLAGS

# Hyperparams
tf.app.flags.DEFINE_string('mode',       'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_string('data_path',  '',      'Path to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '',      'Path to vocabulary file.')
tf.app.flags.DEFINE_string('log_root',   '',      'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name',   '',      'Name for experiment. Logs will be saved in this directory under log_root.')

tf.app.flags.DEFINE_integer('hidden_dim',    256,   'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('batch_size',    16,    'batch size')
tf.app.flags.DEFINE_integer('emb_dim',       128,   'dimension of word embeddings')
tf.app.flags.DEFINE_integer('max_enc_steps', 400,   'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('min_dec_steps', 35,    'Minimum sequence length of generated summary. Applies only for beam search mode')
tf.app.flags.DEFINE_integer('max_dec_steps', 100,   'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('vocab_size',    50000, 'size of vocabulary')
tf.app.flags.DEFINE_integer('beam_size',     4,     'beam size for beam search decoding.')

tf.app.flags.DEFINE_float('lr',                 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc',   0.1,  'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform initializer')
tf.app.flags.DEFINE_float('trun_norm_init_std', 1e-4, 'std of truncated normal initializer')
tf.app.flags.DEFINE_float('max_grad_norm',      2.0,  'for gradient clipping')
tf.app.flags.DEFINE_float('cov_loss_weight',    1.0,  'weight of coverage loss')

tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint. If False (default), run concurrent decoding.')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval dir and save it in the train dir.')
tf.app.flags.DEFINE_boolean('cpu_only', False, 'training with cpu only')
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism.')
tf.app.flags.DEFINE_boolean('convert2coverage', False, 'Convert a non-coverage model to a coverage model.')

def main(unused_args):
    if len(unused_args) != 1: raise Exception('Problem with flags: %s' % unused_args)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting pointer generator in %s mode...', (FLAGS.mode))

    # setup log directory
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)

    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == 'train': os.makedirs(FLAGS.log_root)
        else: raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size

    if FLAGS.single_pass and FLAGS.mode != 'decode':
        raise Exception("The single_pass flag should only be True in decode mode")

    # setup vocabulary
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size, FLAGS.emb_dim)

    # setup hps
    hps_name = ['mode',
                'hidden_dim', 'batch_size', 'emb_dim', 'max_enc_steps', 'max_dec_steps', 'vocab_size',
                'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trun_norm_init_std', 'max_grad_norm', 'cov_loss_weight',
                'cpu_only', 'coverage']
    hps = {}

    for k, v in FLAGS.__flags.items():
        if k in hps_name: hps[k] = v.value

    if FLAGS.mode == 'decode':
        hps['max_dec_steps'] = 1

    hps = namedtuple('HyperParams', hps.keys())(**hps)

    batcher = Batcher(FLAGS.data_path, vocab, hps, FLAGS.single_pass)

    tf.set_random_seed(13131)

    if FLAGS.mode == 'train':
        model = Model(hps)
        run_training(model, batcher)
    elif FLAGS.mode == 'eval':
        model = Model(hps)
        run_eval(model, batcher)
    elif FLAGS.mode == 'decode':
        model = Model(hps)
        decoder = BeamSearchDecoder(model, batcher, vocab)
        decoder.decode()
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

def run_training(model, batcher):

    '''Setup and repeatedly runs training iterations, logging loss to screen and writing summaries'''

    train_dir = os.path.join(FLAGS.log_root, 'train')
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build(device='/cpu:0' if FLAGS.cpu_only else None)

    if FLAGS.convert2coverage:

        assert FLAGS.coverage, 'To convert your model to a coverage model, run with convert_to_coverage=True and coverage=True'

        convert2coverage()

    if FLAGS.restore_best_model:
        restore_best_model()

    try:
        sess_params = {
            'checkpoint_dir': train_dir,
            'config': util.get_config(),
            'hooks': model.sess_hooks
        }

        with tf.train.MonitoredTrainingSession(**sess_params) as sess:
            tf.logging.info('starting run_training')

            while True:
                batch = batcher.next_batch()

                tf.logging.info('running training step...')

                ts = time.time()
                result = model.run_train_step(sess, batch)

                tf.logging.info('seconds for training step: %.3f', time.time() - ts)

                tf.logging.info('loss: %f', result['loss'])

                if FLAGS.coverage:
                    tf.logging.info('coverage loss: %f', result['coverage_loss'])

    except KeyboardInterrupt as e:
        tf.logging.info('Caught keyboard interrupt on worker. Stopping supervisor...')

def restore_best_model():

    '''Load bestmodel file from eval directory, add variables for adagrad, and save to train directory'''

    tf.logging.info('Restoring bestmodel for training...')

    with tf.Session(config=util.get_config()) as sess:
        tf.logging.info('Initializing all variables...')

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver([var for var in tf.all_variables() if 'Adagrad' not in var.name])

        tf.logging.info('Restoring all non-adagrad variables from best model in eval dir...')

        cur_ckpt = util.load_ckpt(sess, saver, 'eval')

        tf.logging.info('Restore %s.' % (cur_ckpt))

        new_model_name = cur_ckpt.split('/')[-1].replace('bestmodel', 'model')
        new_path = os.path.join(FLAGS.log_root, 'train', new_model_name)

        tf.logging.info('Saving model to %s...' % (new_path))

        new_saver = tf.train.Saver()
        new_saver.save(sess, new_path)

        tf.logging.info('Saved.')

        sys.exit()

def convert2coverage():

    '''Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint'''

    tf.logging.info('converting non-coverage model to coverage model...')

    with tf.Session(config=util.get_config) as sess:
        sess.run(tf.global_variables_initializer())

        tf.logging.info('restoring non-coverage variables...')

        saver = tf.train.Saver(var_list=[v for v in tf.global_variables() if 'coverage' not in v.name and 'Adagrad' not in v.name])
        cur_ckpt = util.load_ckpt(sess, saver)
        new_ckpt = cur_ckpt + '_cov_init'

        tf.logging.info('saving model to %s...' % (new_ckpt))

        new_saver = tf.train.Saver()
        new_saver.save(sess, new_ckpt)

    sys.exit()

def run_eval(model, batcher):

    '''Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far.'''

    eval_dir = os.path.join(FLAGS.log_root, 'eval')
    bestmodel_path = os.path.join(eval_dir, 'bestmodel')

    model.build(device='/cpu:0')

    with tf.Session(config=util.get_config()) as sess:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(eval_dir)

        running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
        best_loss = None  # will hold the best loss achieved so far

        while True:
            util.load_ckpt(sess, saver)

            batch = batcher.next_batch()

            ts = time.time()
            result = model.run_eval_step(sess, batch)
            tf.logging.info('seconds for batch: %.2f', time.time() - ts)

            loss = result['loss']
            tf.logging.info('loss: %f', loss)

            if FLAGS.coverage:
                tf.logging.info('coverage loss: %f', result['coverage_loss'])

            summary = result['summary']
            global_step = result['global_step']
            writer.add_summary(summary, global_step)

            running_avg_loss = calc_running_avg_loss(running_avg_loss, np.asscalar(loss), writer, global_step)

            if best_loss is None or running_avg_loss < best_loss:
                tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_path)

                saver.save(sess, bestmodel_path, global_step=global_step, latest_filename='best.ckpt')
                best_loss = running_avg_loss

            if global_step % 100 == 0:
                writer.flush()

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
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    writer.add_summary(loss_sum, step)

    tf.logging.info('running_avg_loss: %f', running_avg_loss)

    return running_avg_loss

if __name__ == '__main__':
    tf.app.run()

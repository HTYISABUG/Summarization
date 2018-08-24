import os, time, logging

import tensorflow as tf
import pyrouge
import numpy as np

import util, data

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60

class BeamSearchDecoder(object):

    """Beam search decoder"""

    def __init__(self, model, batcher, vocab):

        """Initialize decoder.

        Args:
            model: a Seq2SeqAttentionModel object.
            batcher: a Batcher object.
            vocab: Vocabulary object
        """

        self.__model = model
        self.__batcher = batcher
        self.__vocab = vocab

        self.__model.build(device='/cpu:0')

        self.__sess = tf.Session(config=util.get_config())
        self.__saver = tf.train.Saver()

        ckpt_path = util.load_ckpt(self.__sess, self.__saver, )

        if FLAGS.single_pass:
            ckpt_name = 'ckpt-' + ckpt_path.split('-')[-1]

            if 'train' in FLAGS.data_path: dataset = 'train'
            elif 'val' in FLAGS.data_path: dataset = 'val'
            elif 'test' in FLAGS.data_path: dataset = 'test'
            else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))

            dir_name_params = (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
            dir_name = 'decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec' % dir_name_params

            if ckpt_name is not None: dir_name += '_%s' % (ckpt_name)

            self.__dec_dir = os.path.join(FLAGS.log_root, dir_name)

            if os.path.exists(self.__dec_dir):
                raise Exception('single_pass decode directory %s should not already exist' % (self.__dec_dir))
        else:
            self.__dec_dir = os.path.join(FLAGS.log_root, 'decode')

        if not os.path.exists(self.__dec_dir): os.mkdir(self.__dec_dir)

        if FLAGS.single_pass:
            self.__rouge_ref_dir = os.path.join(self.__dec_dir, 'reference')
            self.__rouge_dec_dir = os.path.join(self.__dec_dir, 'decoded')

            if not os.path.exists(self.__rouge_ref_dir):os.mkdir(self.__rouge_ref_dir)
            if not os.path.exists(self.__rouge_dec_dir):os.mkdir(self.__rouge_dec_dir)

    def decode(self):
        cnt = 0

        ts = time.time()

        while True:
            batch = self.__batcher.next_batch()

            if batch is None:
                assert FLAGS.single_pass, 'Dataset exhausted, but we are not in single_pass mode'

                tf.logging.info('Decoder has finished reading dataset for single_pass')

                ref, dec = self.__rouge_ref_dir, self.__rouge_dec_dir

                tf.logging.info('Output has been saved in %s and %s. Now starting ROUGE eval...' % (ref, dec))

                results = rouge_eval(self.__rouge_ref_dir, self.__rouge_dec_dir)
                rouge_log(results, self.__dec_dir)

                return

            origin_article = batch.origin_articles[0]
            origin_abstract = batch.origin_abstracts[0]
            origin_abstract_sens = batch.origin_abstract_sens[0]

            article_with_unks = data.highlight_art_oovs(origin_article, self.__vocab)
            abstract_with_unks = data.highlight_abs_oovs(origin_abstract, self.__vocab, batch.article_oovs[0])

            best_hyp = self.__run(batch)

            dec_ids = [int(t) for t in best_hyp.tokens[1:]]
            dec_words = data.output2words(dec_ids, self.__vocab, batch.article_oovs[0])

            try:
                stop_idx = dec_words.index(data.STOP_TOKEN)
                dec_words = dec_words[:stop_idx]
            except ValueError:
                dec_words = dec_words

            output = ' '.join(dec_words)

            if FLAGS.single_pass:
                self.__write_for_rouge(origin_abstract_sens, dec_words, cnt)
                cnt += 1
            else:
                print()
                tf.logging.info('ARTICLE:  %s', article_with_unks)
                tf.logging.info('REFERENCE SUMMARY: %s', abstract_with_unks)
                tf.logging.info('GENERATED SUMMARY: %s', output)
                print()

                te = time.time()

                if te - ts > SECS_UNTIL_NEW_CKPT:
                    tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint' % (te - ts))
                    util.load_ckpt(self.__sess, self.__saver)
                    ts = time.time()

    def __run(self, batch):
        enc_outputs, dec_in_state = self.__model.run_encoder(self.__sess, batch)

        # Initialize beam_size-many hyptheses
        hyps = [Hypothesis(tokens=[self.__vocab.w2i(data.START_TOKEN)],
                          log_probs=[0.0],
                          state=dec_in_state,
                          attn_dists=[],
                          p_gens=[],
                          coverage=np.zeros([batch.enc_batch.shape[1]]),
                          context_vector=np.zeros([FLAGS.hidden_dim])) for _ in range(FLAGS.beam_size)]

        results = []

        unk_id = self.__vocab.w2i(data.UNKNOWN_TOKEN)
        step = 0

        def sort_hyps(hyps):
            return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)

        while step < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
            latest_tokens = [h.latest_token for h in hyps]
            latest_tokens = [t if t in range(self.__vocab.size) else unk_id for t in latest_tokens]
            states = [h.state for h in hyps]
            prev_coverages = [h.coverage for h in hyps]
            prev_context_vectors = [h.context_vector for h in hyps]

            top_k_ids, top_k_probs, new_states, attn_dists, p_gens, coverages, context_vectors = self.__model.run_decode_once(
                    self.__sess, batch, latest_tokens, enc_outputs, states, prev_coverages, prev_context_vectors)

            new_hyps = []
            num_origin_hyps = 1 if step == 0 else len(hyps)

            for i in range(num_origin_hyps):
                h, new_state, attn_dist, p_gen = hyps[i], new_states[i], attn_dists[i], p_gens[i]
                coverage, context_vector = coverages[i], context_vectors[i]

                for j in range(FLAGS.beam_size * 2):
                    new_hyp = h.next(top_k_ids[i, j], top_k_probs[i, j], new_state, attn_dist, p_gen, coverage, context_vector)
                    new_hyps.append(new_hyp)

            hyps = []

            for h in sort_hyps(new_hyps):
                if h.latest_token == self.__vocab.w2i(data.STOP_TOKEN):

                    if step >= FLAGS.min_dec_steps:
                        results.append(h)
                else:
                    hyps.append(h)

                if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
                    break

            step += 1

        if len(results) == 0:
            results = hyps

        return sort_hyps(results)[0]

    def __write_for_rouge(self, ref_sens, dec_words, flabel):

        '''Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

        Args:
            ref_sens: list of strings
            dec_words: list of strings
            flabel: int, the index with which to label the files
        '''

        dec_sens = []

        while len(dec_words) > 0:
            try:
                period_idx = dec_words.index('.')
            except ValueError:
                period_idx = len(dec_words)

            sen = dec_words[:period_idx+1]
            dec_words = dec_words[period_idx+1:]
            dec_sens.append(' '.join(sen))

        def html_safe(s):

            '''Replace any angled brackets in string s to avoid interfering with HTML attention visualizer.'''

            s.replace('<', '&lt;')
            s.replace('>', '&gt;')

            return s

        ref_sens = [html_safe(s) for s in ref_sens]
        dec_sens = [html_safe(s) for s in dec_sens]

        ref_path = os.path.join(self.__rouge_ref_dir, '%06d_reference.txt' % (flabel))
        dec_path = os.path.join(self.__rouge_dec_dir, '%06d_decoded.txt' % (flabel))

        with open(ref_path, 'w') as fp:
            for i, sen in enumerate(ref_sens):
                fp.write(sen) if i == len(ref_sens) - 1 else fp.write(sen + '\n')

        with open(dec_path, 'w') as fp:
            for i, sen in enumerate(dec_sens):
                fp.write(sen) if i == len(dec_sens) - 1 else fp.write(sen + '\n')

        tf.logging.info("Wrote example %i to file" % flabel)

def rouge_eval(ref_dir, dec_dir):

    '''Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict'''

    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir

    logging.getLogger('global').setLevel(logging.WARNING)

    rouge_results = r.convert_and_evaluate()

    return r.output_to_dict(rouge_results)

def rouge_log(results_dict, dir_to_write):

    """Log ROUGE results to screen and write to file.

    Args:
        results_dict: the dictionary returned by pyrouge
        dir_to_write: the directory where we will write the results to
    """

    log_str = ''

    for x in ['1', '2', 'l']:
        log_str += '\nROUGE-%s:\n' % x

        for y in ['f_score', 'recall', 'precision']:
            key = 'rouge_%s_%s' % (x, y)
            key_cb = key + '_cb'
            key_ce = key + '_ce'

            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]

            log_str += '%s: %.4f with confidence interval (%.4f, %.4f)\n' % (key, val, val_cb, val_ce)

    tf.logging.info(log_str)

    results_file = os.path.join(dir_to_write, 'ROUGE_results.txt')
    tf.logging.info('Writing final ROUGE results to %s...' % (results_file))

    with open(results_file, 'w') as fp:
        fp.write(log_str)

class Hypothesis(object):

    '''Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis.'''

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage, context_vector):

        '''Hypothesis constructor.

        Args:
            tokens: List of integers. The ids of the tokens that form the summary so far.
            log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
            state: Current state of the decoder, a LSTMStateTuple.
            attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length).
            p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model.
            coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
            context_vector: Numpy array of shape (hidden_dim).
        '''

        self.__tokens = tokens
        self.__log_probs = log_probs
        self.__state = state
        self.__attn_dists = attn_dists
        self.__p_gens = p_gens
        self.__coverage = coverage
        self.__context_vector = context_vector

    def next(self, token, log_prob, state, attn_dist, p_gen, coverage, context_vector):

        '''Return a NEW hypothesis, extended with the information from the latest step of beam search.

        Args:
            token: Integer. Latest token produced by beam search.
            log_prob: Float. Log prob of the latest token.
            state: Current decoder state, a LSTMStateTuple.
            attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
            p_gen: Generation probability on latest step. Float.
            coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
            context_vector: Numpy array of shape (hidden_dim).

        Returns:
            New Hypothesis for next step.
        '''

        return Hypothesis(tokens=self.__tokens + [token],
                          log_probs=self.__log_probs + [log_prob],
                          state=state,
                          attn_dists=self.__attn_dists + [attn_dist],
                          p_gens=self.__p_gens + [p_gen],
                          coverage=coverage,
                          context_vector=context_vector)

    @property
    def tokens(self):
        return self.__tokens

    @property
    def state(self):
        return self.__state

    @property
    def latest_token(self):
        return self.__tokens[-1]

    @property
    def log_prob(self):
        return sum(self.__log_probs)

    @property
    def avg_log_prob(self):
        return self.log_prob / len(self.__log_probs)

    @property
    def coverage(self):
        return self.__coverage

    @property
    def context_vector(self):
        return self.__context_vector

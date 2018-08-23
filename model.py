import time
from pprint import pprint

import tensorflow as tf
import numpy as np

class Model(object):

    def __init__(self, hps):
        self.__hps = hps

    def build(self, device=None):
        hps = self.__hps

        tf.logging.info('Building graph...')

        ts = time.time()

        self.__global_step    = tf.train.get_or_create_global_step()
        self.__rand_unif_init = tf.random_normal_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
        self.__trun_norm_init = tf.truncated_normal_initializer(stddev=hps.trun_norm_init_std)

        # encoder
        enc_batch           = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch') # word ids
        enc_lens            = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens') # sentence lengths
        enc_pad_mask        = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_pad_mask') # mask the PAD tokens
        enc_batch_ext_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_ext_vocab')
        self.__max_n_oov    = tf.placeholder(tf.int32, [], name='max_n_oov')

        # decoder
        dec_batch    = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch') # word ids
        dec_pad_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='dec_pad_mask') # mask the PAD tokens
        target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch') # target word ids

        # coverage
        if hps.mode == 'decode' and hps.coverage:
            prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')

        with tf.device(device or '/device:GPU:0'), tf.variable_scope('seq2seq', reuse=tf.AUTO_REUSE):
            enc_inputs, dec_inputs = self.__build_embedding(enc_batch, dec_batch)
            enc_outputs, fw_stat, bw_stat = self.__build_encoder(enc_inputs, enc_lens)
            dec_in_state = self.__build_reduce_states(fw_stat, bw_stat)
            dec_outputs, dec_out_state, attn_dists, p_gens = self.__build_decoder(dec_inputs, enc_outputs, dec_in_state, enc_pad_mask)
            vocab_dists = self.__build_vocab_distribution(dec_outputs)
            final_dists = self.__build_final_distribution(vocab_dists, attn_dists, p_gens, enc_batch_ext_vocab)

            loss = self.__build_loss(final_dists, dec_pad_mask, target_batch)
            train_op = self.__build_train_op(loss)

        if hps.mode == 'decode':
            assert len(final_dists) == 1

            final_dists = final_dists[0]
            top_k_probs, top_k_ids = tf.nn.top_k(final_dists, hps.batch_size * 2)
            top_k_probs = tf.log(top_k_probs)

            self.__top_k_ids = top_k_ids
            self.__top_k_probs = top_k_probs

        self.sess_hooks = [tf.train.NanTensorHook(loss)]

        tf.logging.info('Time to build graph: %i seconds', time.time() - ts)

        # properties for running
        self.__enc_batch           = enc_batch
        self.__enc_lens            = enc_lens
        self.__enc_pad_mask        = enc_pad_mask
        self.__enc_batch_ext_vocab = enc_batch_ext_vocab
        self.__dec_batch           = dec_batch
        self.__dec_pad_mask        = dec_pad_mask
        self.__target_batch        = target_batch

        self.__loss = loss
        self.__train_op = train_op
        self.__summary = tf.summary.merge_all()

        self.__enc_outputs   = enc_outputs
        self.__dec_in_state  = dec_in_state
        self.__dec_out_state = dec_out_state
        self.__attn_dists    = attn_dists
        self.__p_gens        = p_gens

        total_memory = 0

        for var in tf.global_variables():
            memory = np.prod(var.shape) * 4
            total_memory += memory

            # print(var.name, '\t', var.shape, '\t', var.device, '\t', memory, 'Bytes')

        tf.logging.info('Total memory used: %d Bytes' % (total_memory))

    def __build_embedding(self, enc_batch, dec_batch):

        '''Add a word embedding layer to the graph

        Args:
            enc_batch: A tensor of shape [batch_size, <=max_enc_steps].
            dec_batch: A tensor of shape [batch_size, max_dec_steps].

        Returns:
            enc_input_embs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
            dec_input_embs: A tensor of shape [batch_size, max_dec_steps, emb_size].
        '''

        hps = self.__hps

        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=[hps.vocab_size, hps.emb_dim], initializer=self.__trun_norm_init)

            enc_input_embs = tf.nn.embedding_lookup(embedding, enc_batch)
            dec_input_embs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(dec_batch, axis=1)]

            return enc_input_embs, dec_input_embs

    def __build_encoder(self, enc_inputs, enc_lens):

        '''Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
            enc_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
            enc_lens: A tensor of shape [batch_size].

        Returns:
            outputs: A tensor of shape [batch_size, <= max_enc_steps, 2 * hidden_dim].
            fw_state, bw_state: Each are LSTMStateTuples of shape ([batch_size, hidden_dim], [batch_size, hidden_dim])
        '''

        hps = self.__hps

        with tf.variable_scope('encoder'):
            fw = tf.nn.rnn_cell.LSTMCell(hps.hidden_dim, initializer=self.__rand_unif_init, state_is_tuple=True)
            bw = tf.nn.rnn_cell.LSTMCell(hps.hidden_dim, initializer=self.__rand_unif_init, state_is_tuple=True)
            (outputs, (fw_stat, bw_stat)) = tf.nn.bidirectional_dynamic_rnn(fw, bw, enc_inputs,
                                                                            dtype=tf.float32,
                                                                            sequence_length=enc_lens,
                                                                            swap_memory=True)
            outputs = tf.concat(outputs, axis=2)

            return outputs, fw_stat, bw_stat

    def __build_reduce_states(self, fw_stat, bw_stat):

        '''Add a dense layer to reduce the encoder's final state into a single initial state for the decoder.
           This is needed because the encoder is bidirectional but the decoder is not.

        Args:
            fw_stat: LSTMStateTuple with hidden_dim units.
            bw_stat: LSTMStateTuple with hidden_dim units.

        Returns:
            state: LSTMStateTuple with hidden_dim units.
        '''

        hps = self.__hps

        with tf.variable_scope('reduce_states'):
            encoder_c = tf.concat([fw_stat.c, bw_stat.c], axis=1)
            encoder_h = tf.concat([fw_stat.h, bw_stat.h], axis=1)

            decoder_c = tf.layers.dense(encoder_c, hps.hidden_dim,
                                        activation=tf.nn.relu,
                                        kernel_initializer=self.__trun_norm_init,
                                        bias_initializer=self.__trun_norm_init,
                                        name='dec_state_c')

            decoder_h = tf.layers.dense(encoder_h, hps.hidden_dim,
                                        activation=tf.nn.relu,
                                        kernel_initializer=self.__trun_norm_init,
                                        bias_initializer=self.__trun_norm_init,
                                        name='dec_state_h')

            return tf.contrib.rnn.LSTMStateTuple(decoder_c, decoder_h)

    def __build_decoder(self, inputs, enc_outputs, state, enc_pad_mask, prev_coverage=None):

        '''Add attention decoder to the graph.

        Args:
            inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)
            enc_outputs: A tensor of shape [batch_size, <= max_enc_steps, 2 * hidden_dim].
            state: LSTMStateTuple with hidden_dim units.
            enc_pad_mask: A tensor of shape [batch_size, <= max_enc_steps].
            prev_coverage: Previous coverage

        Returns:
            outputs: List of tensors; the outputs of the decoder
            state: The final state of the decoder
            attn_dists: A list containing tensors of shape (batch_size,attn_length).
            p_gens: A list of tensors shape (batch_size, 1); the generation probabilities
            coverage: A tensor, the current coverage
        '''

        hps = self.__hps

        with tf.variable_scope('attention_decoder'):
            cell = tf.nn.rnn_cell.LSTMCell(hps.hidden_dim, initializer=self.__rand_unif_init, state_is_tuple=True)

            with tf.variable_scope('attention'):

                def masked_probability_fn(score):
                    attn_dist = tf.nn.softmax(score)
                    attn_dist *= enc_pad_mask
                    attn_sum  = tf.reduce_sum(attn_dist, axis=1)
                    return attn_dist / tf.reshape(attn_sum, [-1, 1])

                num_units = enc_outputs.get_shape()[2]
                mechanism = CoverageBahdanauAttention(num_units, enc_outputs, probability_fn=masked_probability_fn)
                attention_wrapper = tf.contrib.seq2seq.AttentionWrapper(cell, mechanism, alignment_history=True)

            outputs = []
            attn_dists = []
            p_gens = []

            zero_state = attention_wrapper.zero_state(hps.batch_size, tf.float32)
            state = zero_state.clone(cell_state=state)

            for i, input_ in enumerate(inputs):
                tf.logging.info("Adding attention decoder timestep %i of %i", i + 1, len(inputs))

                context_vector, state = attention_wrapper(input_, state)

                with tf.variable_scope('decoder'):
                    cell_state = state.cell_state
                    output_feature = tf.concat([context_vector, cell_state.c, cell_state.h], axis=1)
                    output = tf.layers.dense(output_feature, hps.hidden_dim, name='output')
                    outputs.append(output)
                    attn_dists.append(state.alignments)

                    p_gen_feature = tf.concat([context_vector, cell_state.c, cell_state.h, input_], axis=1)
                    p_gen = tf.layers.dense(p_gen_feature, 1, activation=tf.nn.sigmoid, name='p_gen')

                    p_gens.append(p_gen)

            return outputs, state.cell_state, attn_dists, p_gens

    def __build_vocab_distribution(self, inputs):
        hps = self.__hps

        with tf.variable_scope('vocab_distribution'):
            return [tf.layers.dense(input_, hps.vocab_size,
                                    activation=tf.nn.softmax,
                                    kernel_initializer=self.__trun_norm_init,
                                    bias_initializer=self.__trun_norm_init,
                                    reuse=tf.AUTO_REUSE) for input_ in inputs]

    def __build_final_distribution(self, vocab_dists, attn_dists, p_gens, enc_batch_ext_vocab):

        '''Calculate the final distribution, for the pointer-generator model

        Args:
            vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
            attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
            p_gens: A list of tensors shape (batch_size, 1); the generation probabilities
            enc_batch_ext_vocab: A tensor of shape [batch_size, None].

        Returns:
            final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        '''

        hps = self.__hps

        with tf.variable_scope('final_distribution'):
            vocab_dists = [p_gen * vocab_dist for p_gen, vocab_dist in zip(p_gens, vocab_dists)]
            attn_dists = [(1 - p_gen) * attn_dist for p_gen, attn_dist in zip(p_gens, attn_dists)]

            ext_vocab_size = hps.vocab_size + self.__max_n_oov
            ext_zeros = tf.zeros([hps.batch_size, self.__max_n_oov])
            ext_vocab_dists = [tf.concat([vocab_dist, ext_zeros], axis=1) for vocab_dist in vocab_dists]

            attn_len = tf.shape(enc_batch_ext_vocab)[1]
            batch_nums = tf.range(hps.batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)
            batch_nums = tf.tile(batch_nums, [1, attn_len])

            indices = tf.stack([batch_nums, enc_batch_ext_vocab], axis=2)
            shape = [hps.batch_size, ext_vocab_size]
            proj_attn_dists = [tf.scatter_nd(indices, attn_dist, shape) for attn_dist in attn_dists]

            return [vocab_dist + attn_dist for vocab_dist, attn_dist in zip(ext_vocab_dists, proj_attn_dists)]

    def __build_loss(self, final_dists, dec_pad_mask, target_batch):
        hps = self.__hps

        with tf.variable_scope('loss'):
            losses = []
            batch_nums = tf.range(hps.batch_size)

            for step, dist in enumerate(final_dists):
                targets = target_batch[:, step]
                indices = tf.stack([batch_nums, targets], axis=1)
                probs = tf.gather_nd(dist, indices)
                loss = -tf.log(probs)
                losses.append(loss)

            dec_lens = tf.reduce_sum(dec_pad_mask, axis=1)
            losses = [loss * dec_pad_mask[:, step] for step, loss in enumerate(losses)]
            loss = tf.reduce_mean(sum(losses) / dec_lens)

            tf.summary.scalar('loss', loss)

            return loss

    def __build_train_op(self, loss):
        hps = self.__hps

        with tf.variable_scope('train_op'):
            grads = tf.gradients(loss, tf.trainable_variables(), aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            grads, global_norm = tf.clip_by_global_norm(grads, hps.max_grad_norm)

            optimizer = tf.train.AdagradOptimizer(hps.lr, initial_accumulator_value=hps.adagrad_init_acc)
            train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()), global_step=self.__global_step)

            tf.summary.scalar('global_norm', global_norm)

            return train_op

    def __make_feed_dict(self, batch, just_enc=False):

        '''Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

        Args:
            batch: Batch object
            just_enc: Boolean. If True, only feed the parts needed for the encoder.
        '''

        feed_dict = {}

        feed_dict[self.__enc_batch]           = batch.enc_batch
        feed_dict[self.__enc_lens]            = batch.enc_lens
        feed_dict[self.__enc_pad_mask]        = batch.enc_pad_mask
        feed_dict[self.__enc_batch_ext_vocab] = batch.enc_batch_ext_vocab
        feed_dict[self.__max_n_oov]           = batch.max_n_oov

        if not just_enc:
            feed_dict[self.__dec_batch]    = batch.dec_batch
            feed_dict[self.__dec_pad_mask] = batch.dec_pad_mask
            feed_dict[self.__target_batch] = batch.target_batch

        return feed_dict

    def run_train_step(self, sess, batch):

        '''Runs one training iteration. Returns a dict containing train op, summaries, loss, global_step.'''

        feed_dict = self.__make_feed_dict(batch)
        rets = {'global_step': self.__global_step,
                'loss':        self.__loss,
                'train_op':    self.__train_op,}

        return sess.run(rets, feed_dict=feed_dict)

    def run_eval_step(self, sess, batch):
        feed_dict = self.__make_feed_dict(batch)
        rets = {'global_step': self.__global_step,
                'loss':        self.__loss,
                'summary':     self.__summary,}

        return sess.run(rets, feed_dict=feed_dict)

    def run_encoder(self, sess, batch):

        '''For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

        Args:
            sess: Tensorflow session.
            batch: Batch object that is the same example repeated across the batch (for beam search)

        Returns:
            enc_outputs: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
            dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
        '''

        feed_dict = self.__make_feed_dict(batch, just_enc=True)
        enc_outputs, dec_in_state, global_step = sess.run(
            [self.__enc_outputs, self.__dec_in_state, self.__global_step],
            feed_dict=feed_dict
        )

        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])

        return enc_outputs, dec_in_state

    def run_decode_once(self, sess, batch, latest_tokens, enc_states, dec_in_states):

        '''For beam search decoding. Run the decoder for one step.

        Args:
            sess: Tensorflow session.
            batch: Batch object containing single example repeated across the batch
            latest_tokens: Tokens to be fed as input into the decoder for this timestep
            enc_states: The encoder states.
            dec_in_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep

        Returns:
            ids: top 2k ids. shape [beam_size, 2*beam_size]
            probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
            dec_out_states: a list length beam_size containing LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
            attn_dists: List length beam_size containing lists length attn_length.
            p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
        '''

        beam_size = len(dec_in_states)

        new_c = np.concatenate([np.expand_dims(state.c, axis=0) for state in dec_in_states], axis=0)
        new_h = np.concatenate([np.expand_dims(state.h, axis=0) for state in dec_in_states], axis=0)
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed_dict = {
            self.__enc_pad_mask: batch.enc_pad_mask,
            self.__enc_batch_ext_vocab: batch.enc_batch_ext_vocab,
            self.__max_n_oov: batch.max_n_oov,
            self.__enc_outputs: enc_states,
            self.__dec_batch: np.transpose([latest_tokens]),
            self.__dec_in_state: new_dec_in_state
        }

        rets = {
            'ids':        self.__top_k_ids,
            'probs':      self.__top_k_probs,
            'states':     self.__dec_out_state,
            'attn_dists': self.__attn_dists,
            'p_gens':     self.__p_gens
        }

        result = sess.run(rets, feed_dict=feed_dict)

        assert len(result['attn_dists']) == 1
        assert len(result['p_gens']) == 1

        dec_out_states = [tf.nn.rnn_cell.LSTMStateTuple(result['states'].c[i], result['states'].h[i]) for i in range(beam_size)]
        attn_dists = result['attn_dists'][0].tolist()
        p_gens = result['p_gens'][0].tolist()

        return result['ids'], result['probs'], dec_out_states, attn_dists, p_gens

class CoverageBahdanauAttention(tf.contrib.seq2seq.BahdanauAttention):

    '''Bahdanau-style attention with coverage.'''

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name='BahdanauAttention'):

        super(CoverageBahdanauAttention, self).__init__(num_units,
                                                        memory,
                                                        memory_sequence_length=memory_sequence_length,
                                                        normalize=False,
                                                        probability_fn=probability_fn,
                                                        score_mask_value=score_mask_value,
                                                        dtype=dtype,
                                                        name=name)

    def __call__(self, query, state):

        with tf.variable_scope(None, 'bahdanau_attention', [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = self.__bahdanau_score(processed_query, self._keys)

        alignments = self._probability_fn(score, state)
        next_state = alignments

        return alignments, next_state

    def __bahdanau_score(self, processed_query, keys):
        dtype = processed_query.dtype

        # Get the number of hidden units from the trailing dimension of keys
        num_units = keys.shape[2].value or tf.shape(keys)[2]

        # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
        processed_query = tf.expand_dims(processed_query, 1)

        v = tf.get_variable('attention_v', [num_units], dtype=dtype)

        return tf.reduce_sum(v * tf.tanh(keys + processed_query), [2])

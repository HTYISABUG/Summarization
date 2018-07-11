import tensorflow as tf

class Model(object):

    def __init__(self, hps):
        self.__hps = hps

    def build(self):
        hps = self.__hps

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # encoder
        self.__enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch') # word ids
        self.__enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens') # sentence lengths
        self.__enc_pad_mask = tf.placeholder(tf.bool, [hps.batch_size, None], name='enc_pad_mask') # mask the PAD tokens
        self.__enc_batch_ext_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_ext_vocab')
        self.__max_n_oov = tf.placeholder(tf.int32, [], name='max_n_oov')

        # decoder
        self.__dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch') # word ids
        self.__dec_pad_mask = tf.placeholder(tf.bool, [hps.batch_size, None], name='dec_pad_mask') # mask the PAD tokens
        self.__target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch') # target word ids

        # embedding
        self.__embs = tf.placeholder(tf.float32, [hps.vocab_size, hps.emb_dim], name='embs')

        self.__build_seq2seq()

        self.__summary = tf.summary.merge_all()

    def __build_seq2seq(self):
        hps = self.__hps

        with tf.variable_scope('seq2seq', reuse=tf.AUTO_REUSE):
            self.rand_unif_init = tf.random_normal_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
            self.trun_norm_init = tf.truncated_normal_initializer(stddev=hps.trun_norm_init_std)

            enc_inputs, dec_inputs = self.__build_embedding()
            enc_outputs, fw_stat, bw_stat = self.__build_encoder(enc_inputs)
            dec_in_state = self.__build_reduce_states(fw_stat, bw_stat)
            dec_outputs, dec_out_state, p_gens = self.__build_decoder(dec_inputs, enc_outputs, dec_in_state)
            vocab_dists = self.__build_vocab_distribution(dec_outputs)
            final_dists = self.__build_final_distribution(vocab_dists, dec_out_state.attention_state.stack(), p_gens)

            loss = self.__build_loss(final_dists)
            train_op = self.__build_train_op(loss)

        if hps.mode == 'decode':
            assert len(final_dists) == 1

            final_dists = final_dists[0]
            top_k_probs, top_k_ids = tf.nn.top_k(final_dists, hps.batch_size * 2)
            top_k_probs = tf.log(top_k_probs)

    def __build_embedding(self):
        hps = self.__hps

        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=[hps.vocab_size, hps.emb_dim], dtype=tf.float32, trainable=False)
            self.__load_embs = tf.assign(embedding, self.__embs)

            enc_input_embs = tf.nn.embedding_lookup(embedding, self.__enc_batch)
            dec_input_embs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self.__dec_batch, axis=1)]

            return enc_input_embs, dec_input_embs

    def __build_encoder(self, inputs):

        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
            encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].

        Returns:
            outputs: A tensor of shape [batch_size, <= max_enc_steps, 2 * hidden_dim].
            fw_state, bw_state: Each are LSTMStateTuples of shape ([batch_size, hidden_dim], [batch_size, hidden_dim])
        """

        hps = self.__hps

        with tf.variable_scope('encoder'):
            fw = tf.nn.rnn_cell.LSTMCell(hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            bw = tf.nn.rnn_cell.LSTMCell(hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            (outputs, (fw_stat, bw_stat)) = tf.nn.bidirectional_dynamic_rnn(fw, bw, inputs, dtype=tf.float32, sequence_length=self.__enc_lens, swap_memory=True)
            outputs = tf.concat(outputs, axis=2)

            return outputs, fw_stat, bw_stat

    def __build_reduce_states(self, fw_stat, bw_stat):

        """Add a dense layer to reduce the encoder's final state into a single initial state for the decoder.
           This is needed because the encoder is bidirectional but the decoder is not.

        Args:
            fw_stat: LSTMStateTuple with hidden_dim units.
            bw_stat: LSTMStateTuple with hidden_dim units.

        Returns:
            state: LSTMStateTuple with hidden_dim units.
        """

        hps = self.__hps

        with tf.variable_scope('reduce_states'):
            encoder_c = tf.concat([fw_stat.c, bw_stat.c], axis=1)
            encoder_h = tf.concat([fw_stat.h, bw_stat.h], axis=1)

            decoder_c = tf.layers.dense(encoder_c, hps.hidden_dim, activation=tf.nn.relu)
            decoder_h = tf.layers.dense(encoder_h, hps.hidden_dim, activation=tf.nn.relu)

            return tf.contrib.rnn.LSTMStateTuple(decoder_c, decoder_h)

    def __build_decoder(self, inputs, enc_outputs, state):

        """Add attention decoder to the graph.

        Args:
            inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)
            enc_outputs: A tensor of shape [batch_size, <= max_enc_steps, 2 * hidden_dim].
            state: LSTMStateTuple with hidden_dim units.

        Returns:
            outputs: List of tensors; the outputs of the decoder
            state: The final state of the decoder
            p_gens: A list of tensors shape (batch_size, 1); the generation probabilities
        """

        hps = self.__hps

        with tf.variable_scope('decoder'):
            cell = tf.nn.rnn_cell.LSTMCell(hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)

            with tf.variable_scope('attention'):

                def masked_probability_fn(score):
                    attn_dist = tf.nn.softmax(score)
                    attn_dist *= self.__enc_pad_mask
                    attn_sum  = tf.reduce_sum(attn_dist, axis=1)
                    return attn_dist / tf.reshape(attn_sum, [-1, 1])

                num_units = self.__enc_outputs.get_shape()[2]
                mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, enc_outputs, probability_fn=masked_probability_fn)
                attention_wrapper = tf.contrib.seq2seq.AttentionWrapper(cell, mechanism, alignment_history=True)

            outputs = []
            p_gens = []

            for input_ in inputs:
                context_vector, state = attention_wrapper(input_, state)

                output = tf.layers.dense(tf.concat([context_vector, state], axis=1), hps.hidden_dim)
                outputs.append(output)

                with tf.variable_scope('p_gen'):
                    p_gen_feature = tf.concat([context_vector, state.c, state.h, input_], axis=1)
                    p_gen = tf.layers.dense(p_gen_feature, 1, activation=tf.nn.softmax)

                p_gens.append(p_gen)

            return outputs, state, p_gens

    def __build_vocab_distribution(self, inputs):
        hps = self.__hps

        with tf.variable_scope('vocab_distribution'):
            return [tf.layers.dense(input_, hps.vocab_size, activation=tf.nn.softmax) for input_ in inputs]

    def __build_final_distribution(self, vocab_dists, attn_dists, p_gens):

        """Calculate the final distribution, for the pointer-generator model

        Args:
            vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
            attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
            p_gens: A list of tensors shape (batch_size, 1); the generation probabilities

        Returns:
            final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """

        hps = self.__hps

        with tf.variable_scope('final_distribution'):
            vocab_dists = [p_gen * vocab_dist for p_gen, vocab_dist in zip(p_gens, vocab_dists)]
            attn_dists = [(1 - p_gen) * attn_dist for p_gen, attn_dist in zip(p_gens, attn_dists)]

            ext_vocab_size = hps.vocab_size + self.__max_n_oov
            ext_zeros = tf.zeros([hps.batch_size, self.__max_n_oov])
            ext_vocab_dists = [tf.concat([vocab_dist, ext_zeros], axis=1) for vocab_dist in vocab_dists]

            attn_len = tf.shape(self.__enc_batch_ext_vocab)[1]
            batch_nums = tf.range(hps.batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)
            batch_nums = tf.tile(batch_nums, [1, attn_len])

            indices = tf.stack([batch_nums, self.__enc_batch_ext_vocab], axis=2)
            shape = [hps.batch_size, ext_vocab_size]
            proj_attn_dists = [tf.scatter_nd(indices, attn_dist, shape) for attn_dist in attn_dists]

            return [vocab_dist + attn_dist for vocab_dist, attn_dist in zip(vocab_dists, proj_attn_dists)]

    def __build_loss(self, final_dists):
        hps = self.__hps

        with tf.variable_scope('loss'):
            losses = []
            batch_nums = tf.range(hps.batch_size)

            for step, dist in enumerate(final_dists):
                targets = self.__target_batch[:, step]
                indices = tf.stack([batch_nums, targets], axis=1)
                probs = tf.gather_nd(dist, indices)
                loss = -tf.log(probs)
                losses.append(loss)

            dec_lens = tf.reduce_sum(self.__dec_pad_mask, axis=1)
            losses = [loss * self.__dec_pad_mask[:, step] for step, loss in enumerate(losses)]
            loss = tf.reduce_mean(sum(losses) / dec_lens)

            tf.summary.scalar('loss', loss)

            return loss

    def __build_train_op(self, loss):
        hps = self.__hps

        with tf.variable_scope('train_op'):
            grads = tf.gradients(loss, tf.trainable_variables(), aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            grads, global_norm = tf.clip_by_global_norm(grads, hps.max_grad_norm)

            tf.summary.scalar('global_norm', global_norm)

            optimizer = tf.train.AdagradOptimizer(hps.lr, initial_accumulator_value=hps.adagrad_init_acc)
            return optimizer.apply_gradients(zip(grads, tf.trainable_variables), global_step=self.__global_step, name='train_op')

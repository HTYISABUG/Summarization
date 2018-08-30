import queue, random, time
from threading import Thread

import numpy as np
import tensorflow as tf

import data

class Batcher(object):

    BATCH_QUEUE_MAX = 100 # Max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, hps, single_pass):

        """Initialize the batcher. Start threads that process the data into batches.

        Args:
            data_path: tf.Example filepattern.
            vocab: Vocabulary object
            hps: hyperparameters
            single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
        """

        self.__data_path = data_path
        self.__vocab = vocab
        self.__hps = hps
        self.__single_pass = single_pass

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self.__example_queue = queue.Queue(self.BATCH_QUEUE_MAX * hps.batch_size)
        self.__batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self.__n_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self.__n_batch_q_threads = 1    # just one thread to batch examples
            self.__bucketing_cache_size = 1 # only load one batches-worth of examples before bucketing
            self.__finished = False         # this will tell us when we're finished reading the dataset
        else:
            self.__n_example_q_threads = 16     # num threads to fill example queue
            self.__n_batch_q_threads = 4        # num threads to fill batch queue
            self.__bucketing_cache_size = 100   # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self.__example_q_threads = []
        self.__batch_q_threads = []

        for _ in range(self.__n_example_q_threads):
            self.__example_q_threads.append(Thread(target=self.__fill_example_queue))
            self.__example_q_threads[-1].daemon = True
            self.__example_q_threads[-1].start()

        for _ in range(self.__n_batch_q_threads):
            self.__batch_q_threads.append(Thread(target=self.__fill_batch_queue))
            self.__batch_q_threads[-1].daemon = True
            self.__batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:
            self.__watch_thread = Thread(target=self.__watch_threads)
            self.__watch_thread.daemon = True
            self.__watch_thread.start()

    def __fill_example_queue(self):

        """Reads data from file and processes into Examples which are then placed into the example queue."""

        text_gen = self.__text_generator(data.tf_example_generator(self.__data_path, self.__single_pass))

        while True:
            try:
                article, abstract = next(text_gen)
            except StopIteration:
                tf.logging.info("The tf example generator for this example queue filling thread has exhausted data.")

                if self.__single_pass:
                    tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self.__finished = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            abstract_sens = [sen.strip() for sen in data.abstract2sens(abstract)]
            example = Example(article, abstract_sens, self.__vocab, self.__hps)
            self.__example_queue.put(example)

    def __text_generator(self, tf_example_generator):
        while True:
            e = next(tf_example_generator)

            try:
                article = e.features.feature['article'].bytes_list.value[0].decode()
                abstract = e.features.feature['abstract'].bytes_list.value[0].decode()
            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue

            if len(article) == 0:
                tf.logging.warning('Found an example with empty article text. Skipping it.')
            else:
                yield article, abstract

    def __fill_batch_queue(self):

        """Takes Examples out of queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

        In decode mode, makes batches that each contain a single example repeated.
        """

        hps = self.__hps

        while True:
            if hps.mode != 'decode':
                # Get examples then sort
                inputs = [self.__example_queue.get() for _ in range(hps.batch_size * self.__bucketing_cache_size)]
                inputs.sort(key=lambda input_: input_.enc_len)

                # Group the sorted examples in to batches, and push into batch queue
                batches = [inputs[i:i+hps.batch_size] for i in range(0, len(inputs), hps.batch_size)]

                if not self.__single_pass:
                    random.shuffle(batches)

                for b in batches:
                    self.__batch_queue.put(Batch(b, self.__vocab, hps))
            else:
                e = self.__example_queue.get()
                b = [e for _ in range(hps.batch_size)]
                self.__batch_queue.put(Batch(b, self.__vocab, hps))

    def __watch_threads(self):

        """Watch example queue and batch queue threads and restart if dead."""

        while True:
            time.sleep(60)

            for i, t in enumerate(self.__example_q_threads):
                if not t.isAlive():
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    nt = Thread(target=self.__fill_example_queue)
                    self.__example_q_threads[i] = nt
                    nt.daemon=True
                    nt.start()

            for i, t in enumerate(self.__batch_q_threads):
                if not t.isAlive():
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    nt = Thread(target=self.__fill_batch_queue)
                    self.__batch_q_threads[i] = nt
                    nt.daemon = True
                    nt.start()

    def next_batch(self):
        if self.__batch_queue.qsize() == 0:
            tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self.__batch_queue.qsize(), self.__example_queue.qsize())

            if self.__single_pass and self.__finished:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        return self.__batch_queue.get()

class Example(object):

    """Class representing a train/val/test example for text summarization."""

    def __init__(self, article, abstract_sens, vocab, hps):

        """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences.

        Args:
            article: source text; a string. each token is separated by a single space.
            abstract_sens: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
            vocab: Vocabulary object
            hps: hyperparameters
        """

        self.__hps = hps

        start_id = vocab.w2i(data.START_TOKEN)
        stop_id = vocab.w2i(data.STOP_TOKEN)

        # article
        article_words = article.split()

        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]

        self.enc_len = len(article_words)
        self.enc_input = [vocab.w2i(w) for w in article_words]

        # abstract
        abstract = ' '.join(abstract_sens)
        abstract_words = abstract.split()

        if len(abstract_words) > hps.max_dec_steps:
            abstract_words = abstract_words[:hps.max_dec_steps]

        # pointer generator

        ## Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id and the in-article OOVs words
        self.enc_input_ext_vocab, self.article_oovs = data.article2ids(article_words, vocab)

        ## Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        abs_ids = [vocab.w2i(w) for w in abstract_words]

        self.dec_input, self.dec_target = self.__get_dec_input_target_seqs(abs_ids, hps.max_dec_steps, start_id, stop_id)
        self.dec_len = len(self.dec_input)

        abs_ids_ext_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)
        _, self.ext_dec_target = self.__get_dec_input_target_seqs(abs_ids_ext_vocab, hps.max_dec_steps, start_id, stop_id)

        # origin backup
        self.origin_article = article
        self.origin_abstract = abstract
        self.origin_abstract_sens = abstract_sens

    def __get_dec_input_target_seqs(self, sequence, max_len, start_id, stop_id):
        input_ = [start_id] + sequence
        target = sequence[:]

        if len(input_) > max_len:
            input_ = input_[:max_len]
            target = target[:max_len]
        else:
            target.append(stop_id)

        assert len(input_) == len(target)

        return input_, target

    def pad_enc_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)

        while len(self.enc_input_ext_vocab) < max_len:
            self.enc_input_ext_vocab.append(pad_id)

    def pad_dec_input_target(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)

        while len(self.dec_target) < max_len:
            self.dec_target.append(pad_id)

        while len(self.ext_dec_target) < max_len:
            self.ext_dec_target.append(pad_id)

class Batch(object):

    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, examples, vocab, hps):
        self.__pad_id = vocab.w2i(data.PAD_TOKEN)

        self.__init_enc_seq(examples, hps)
        self.__init_dec_seq(examples, hps)
        self.__store_origin(examples)

    def __init_enc_seq(self, examples, hps):

        """Initializes the following:

            self.enc_batch:
                numpy array of shape (batch_size, <=max_enc_steps) containing integer ids, padded to length of longest seq in the batch
            self.enc_lens:
                numpy array of shape (batch_size) containing integers. The length of each encoder input sequence (pre-padding).
            self.enc_padding_mask:
                numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s.
            self.max_art_oovs:
                maximum number of in-article OOVs in the batch
            self.art_oovs:
                list of list of in-article OOVs (strings), for each example in the batch
            self.enc_batch_extend_vocab:
                Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
        """

        max_enc_len = max([e.enc_len for e in examples])

        for e in examples:
            e.pad_enc_input(max_enc_len, self.__pad_id)

        # Initialize the numpy arrays
        self.enc_batch = np.zeros([hps.batch_size, max_enc_len], dtype=np.int32)
        self.enc_lens = np.zeros([hps.batch_size], dtype=np.int32)
        self.enc_pad_mask = np.zeros([hps.batch_size, max_enc_len], dtype=np.float32)
        self.enc_batch_ext_vocab = np.zeros([hps.batch_size, max_enc_len], dtype=np.int32)

        # Fill in the numpy arrays
        for i, e in enumerate(examples):
            self.enc_batch[i] = e.enc_input
            self.enc_lens[i] = e.enc_len
            self.enc_pad_mask[i][:e.enc_len] = 1
            self.enc_batch_ext_vocab[i] = e.enc_input_ext_vocab

        self.max_n_oov = max([len(e.article_oovs) for e in examples])
        self.article_oovs = [e.article_oovs for e in examples]

    def __init_dec_seq(self, examples, hps):

        """Initializes the following:

            self.dec_batch:
                numpy array of shape (batch_size, max_dec_steps), containing integer ids, padded to max_dec_steps length.
            self.target_batch:
                numpy array of shape (batch_size, max_dec_steps), containing integer ids, padded to max_dec_steps length.
            self.dec_padding_mask:
                numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s.
        """

        for e in examples:
            e.pad_dec_input_target(hps.max_dec_steps, self.__pad_id)

        self.dec_batch = np.zeros([hps.batch_size, hps.max_dec_steps], dtype=np.int32)
        self.dec_pad_mask = np.zeros([hps.batch_size, hps.max_dec_steps], dtype=np.float32)
        self.target_batch = np.zeros([hps.batch_size, hps.max_dec_steps], dtype=np.int32)
        self.ext_target_batch = np.zeros([hps.batch_size, hps.max_dec_steps], dtype=np.int32)

        for i, e in enumerate(examples):
            self.dec_batch[i] = e.dec_input
            self.dec_pad_mask[i][:e.dec_len] = 1
            self.target_batch[i] = e.dec_target
            self.ext_target_batch[i] = e.ext_dec_target

    def __store_origin(self, examples):
        self.origin_articles = [e.origin_article for e in examples]
        self.origin_abstracts = [e.origin_abstract for e in examples]
        self.origin_abstract_sens = [e.origin_abstract_sens for e in examples]

import glob, random, struct

import tensorflow as tf
import numpy as np

from gensim.models import KeyedVectors

_SENTENCE_START = '<s>'
_SENTENCE_END   = '</s>'

PAD_TOKEN       = 'PAD'
UNKNOWN_TOKEN   = 'UNK'
START_TOKEN     = 'START'
STOP_TOKEN      = 'STOP'

class Vocab(object):

    def __init__(self, vocab_file, vocab_size, emb_dim):
        self.__word2id = {}
        self.__id2word = {}

        cnt = 0

        wv = KeyedVectors.load_word2vec_format('./word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        wv_dim = wv[PAD_TOKEN].shape[0]

        self.embedding = np.zeros((vocab_size, emb_dim), dtype=np.float32)

        for w in [PAD_TOKEN, UNKNOWN_TOKEN, START_TOKEN, STOP_TOKEN]:
            self.__word2id[w] = cnt
            self.__id2word[cnt] = w

            if wv_dim < emb_dim:
                self.embedding[cnt][:wv_dim] = wv[w]
            else:
                self.embedding[cnt] = wv[w][:emb_dim]

            cnt += 1

        with open(vocab_file, 'r') as fp:
            for line in fp:
                pieces = line.split()

                if len(pieces) != 2:
                    print('Warning: skip incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue

                w = pieces[0]

                if w in [_SENTENCE_START, _SENTENCE_END, PAD_TOKEN, UNKNOWN_TOKEN, START_TOKEN, STOP_TOKEN]:
                    raise Exception('<s>, </s>, UNK, PAD, START and STOP shouldn\'t be in the vocab file, but %s is' % w)
                elif w in self.__word2id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                elif w not in wv.vocab:
                    continue

                self.__word2id[w] = cnt
                self.__id2word[cnt] = w

                if wv_dim < emb_dim:
                    self.embedding[cnt][:wv_dim] = wv[w]
                else:
                    self.embedding[cnt] = wv[w][:emb_dim]

                cnt += 1

                if vocab_size != 0 and cnt >= vocab_size:
                    print('size of vocab was specified as %i; we now have %i words. Stopping reading.' % (vocab_size, cnt))
                    break

            self.__size = cnt
            self.embedding = self.embedding[:cnt]

            print('Finished constructing vocabulary of %i total words. Last word added: %s' % (cnt, self.__id2word[cnt-1]))

    def w2i(self, w):
        return self.__word2id[w if w in self.__word2id else UNKNOWN_TOKEN]

    def i2w(self, i):
        if i not in self.__id2word:
            raise ValueError('Id not found in vocab: %d' % i)
        return self._id_to_word[word_id]

    def size(self):
        return self.__size

def article2ids(article_words, vocab):
    """Map the article words to their ids. Also return a list of OOVs in the article.

    Args:
        article_words: list of words (strings)
        vocab: Vocabulary object

    Returns:
        ids: A list of word ids (integers); OOVs are represented by their temporary article OOV number.
        oovs: A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers.
    """

    ids = []
    oovs = []
    unk_id = vocab.w2i(UNKNOWN_TOKEN)

    for w in article_words:
        id_ = vocab.w2i(w)

        if id_ == unk_id:
            if w not in oovs:
                oovs.append(w)

            oov_idx = oovs.index(w)
            ids.append(vocab.size() + oov_idx)
        else:
            ids.append(id_)

    return ids, oovs

def abstract2ids(abstract_words, vocab, article_oovs):
    """ Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.

    Args:
        abstract_words: list of words (strings)
        vocab: Vocabulary object
        article_oovs: list of in-article OOV words (strings), in the order corresponding to their temporary article OOV numbers

    Returns:
        ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id.
    """

    ids = []
    unk_id = vocab.w2i(UNKNOWN_TOKEN)

    for w in abstract_words:
        i = vocab.w2i(w)

        if i == unk_id: # if w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                tid = vocab.size() + article_oovs.index(w)
                ids.append(tid)
            else: # If w is an out-of-article OOV
                ids.append(unk_id)
        else:
            ids.append(i)

    return ids

def output2words(ids, vocab, article_oovs):
    """ Maps output ids to words, including mapping in-article OOVs from their temporary ids to the original OOV string (applicable in pointer-generator mode).

    Args:
        ids: list of ids (integers)
        vocab: Vocabulary object
        article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids

    Returns:
        words: list of words (strings)
    """

    words = []

    for i in ids:
        try:
            w = vocab.i2w(i)
        except ValueError as e:
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary."
            article_oovs_idx = i - vocab.size()

            try:
                w = article_oovs[article_oovs_idx]
            except ValueError as e:
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))

        words.append(w)

    return words

def abstract2sens(abstract):
    """ Splits abstract text from datafile into list of sentences.

    Args:
        abstract: string containing <s> and </s> tags for starts and ends of sentences

    Returns:
        sens: List of sentence strings (no tags)
    """

    cnt = 0
    sens = []

    while True:
        try:
            start = abstract.index(_SENTENCE_START, cnt)
            end = abstract.index(_SENTENCE_END, start + 1)
            cnt = end + len(_SENTENCE_END)
            sens.append(abstract[start+len(_SENTENCE_START):end])
        except ValueError as e:
            return sens

def tf_example_generator(data_path, single_pass):
    """Generates tf.Examples from data files.

        Binary data format: <length><blob>. <length> represents the byte size of <blob>.
        <blob> is serialized tf.Example proto.
        The tf.Example contains the tokenized article text and summary.

    Args:
        data_path: Path to tf.Example data files. Can include wildcards.
        single_pass: Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.

    Yields:
        Deserialized tf.Example.
    """

    while True:
        filelist = glob.glob(data_path) # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty

        if single_pass: filelist = sorted(filelist)
        else: random.shuffle(filelist)

        for f in filelist:
            with open(f, 'rb') as fp:
                while True:
                    str_len = fp.read(8) # paper write string length with type long long (size 8 bytes)
                    if not str_len: break
                    str_len = struct.unpack('q', str_len)[0]
                    example_str = struct.unpack('%ds' % (str_len), fp.read(str_len))[0]
                    yield tf.train.Example.FromString(example_str)

        if single_pass:
            print('tf_example_generator completed reading all datafiles. No more data.')
            break
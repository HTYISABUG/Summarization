import glob, random, struct

import tensorflow as tf

_SENTENCE_START = '<s>'
_SENTENCE_END   = '</s>'

_PAD_TOKEN       = 'PAD'
_UNKNOWN_TOKEN   = 'UNK'
_START_TOKEN     = 'START'
_STOP_TOKEN      = 'STOP'

class Vocab(object):

    def __init__(self, vocab_file, vocab_size):
        self.__word2id = {}
        self.__id2word = {}

        cnt = 0

        for w in [_PAD_TOKEN, _UNKNOWN_TOKEN, _START_TOKEN, _STOP_TOKEN]:
            self.__word2id[w] = cnt
            self.__id2word[cnt] = w
            cnt += 1

        with open(vocab_file, 'r') as fp:
            for line in fp:
                pieces = line.split()

                if len(pieces) != 2: print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)

                w = pieces[0]

                if w in [_SENTENCE_START, _SENTENCE_END, _PAD_TOKEN, _UNKNOWN_TOKEN, _START_TOKEN, _STOP_TOKEN]:
                    raise Exception('<s>, </s>, UNK, PAD, START and STOP shouldn\'t be in the vocab file, but %s is' % w)
                elif w in self.__word2id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)

                self.__word2id[w] = cnt
                self.__id2word[cnt] = w
                cnt += 1

                if max_size != 0 and self._count >= max_size:
                    print('max_size of vocab was specified as %i; we now have %i words. Stopping reading.' % (vocab_size, cnt))
                    break

            self.__size = cnt

            print('Finished constructing vocabulary of %i total words. Last word added: %s') % (cnt, self.__id2word[cnt-1])

    def w2i(self, w):
        return self.__word2id[w if w in self.__word2id else _UNKNOWN_TOKEN]

    def i2w(self, i):
        if i not in self.__id2word:
            raise ValueError('Id not found in vocab: %d' % i)
        return self._id_to_word[word_id]

    def size(self):
        return self.__size

def abstract2id(abstract_word, vocab, article_oovs):
    """ Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.

    Args:
        abstract_word: list of words (strings)
        vocab: Vocabulary object
        article_oovs: list of in-article OOV words (strings), in the order corresponding to their temporary article OOV numbers

    Returns:
        ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id.
    """

    ids = []
    unk_id = vocab.w2i(_UNKNOWN_TOKEN)

    for w in abstract_word:
        i = vocab.w2i[w]

        if i == unk_id: # if w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                tid = vocab.size() + article_oovs.index(w)
                ids.append(tid)
            else: # If w is an out-of-article OOV
                ids.append(unk_id)
        else:
            ids.append(i)

    return ids

def output2word(self, ids, vocab, article_oovs):
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

def abstract2sen(self, abstract):
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
                    str_len = reader.read(8) # paper write string length with type long long (size 8 bytes)
                    if not byte_length: break
                    str_len = struct.unpack('q', str_len)[0]
                    example_str = struct.unpack('%ds' % (str_len), reader.read(str_len))[0]
                    yield tf.train.Example.FromString(example_str)

        if single_pass:
            print('tf_example_generator completed reading all datafiles. No more data.')
            break

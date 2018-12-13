import jaconv
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Vocab:

    sos = '<sos>'
    eos = '<eos>'
    pad = '<pad>'
    word_not_found = '<404>'
    id_not_found = -1111

    def __init__(self, dict_of_unique_words):
        word2id = {}
        id2word = {}

        # :list_of_unique_words: dict <word>:<frequency>
        list_of_unique_words = sorted(dict_of_unique_words.items(), key=lambda _: _[1])

        # adding some specified tokens
        word2id[Vocab.sos] = 0
        word2id[Vocab.eos] = 1
        word2id[Vocab.pad] = 2

        _offset = len(Vocab.word2id)

        word2id = dict([(word, freq + _offset) for (word, freq) in enumerate(list_of_unique_words)])
        id2word = dict([(freq, word) for (freq, word) in Vocab.word2id.items()])

        self.word2id = word2id
        self.id2word = id2word

    def get_id_from_word(self, word):
        return Vocab.id_not_found if word not in self.word2id else self.word2id[word]

    def get_word_from_id(self, id):
        return Vocab.word_not_found if id not in self.id2word else self.id2word[id]

class Config:
    image_folder = ""
    image_fn_train = ""
    label_fn_train = ""
    min_freq = 1

    image_fn_valid = ""
    label_fn_valid = ""

    @classmethod
    def set(cls, args):
        Config.image_folder = args.get("image_folder", Config.image_folder)
        Config.image_fn_train = args.get("image_fn_train", Config.image_fn_train)
        Config.label_fn_train = args.get("label_fn_train", Config.label_fn_train)
        Config.min_freq = args.get("min_freq",Config.min_freq)

        Config.image_fn_valid = args.get("image_fn_valid",Config.image_fn_valid)
        Config.label_fn_valid = args.get("label_fn_valid",Config.label_fn_valid)

class TextUtils:
    @classmethod
    def normalize(cls, text):
        text = jaconv.normalize(text=text.strip())

        return text

    @classmethod
    def get_dict_of_unique_words(cls, text_lines, normalize=True, min_freq=1):
        def _filter_by_freq(_dct, _min_freq):
            return dict([(__dct, __freq) for (__dct, __freq) in _dct.items() if __freq >= _min_freq])

        dct = {}

        for text_line in text_lines:
            if normalize: text_line = TextUtils.normalize(text_line)
            for c in text_line:
                if c not in dct: dct[c] = 1
                else: dct[c] += 1

        return _filter_by_freq(dct, _min_freq=min_freq)

    @classmethod
    def to_ids_from_sentence(self, sentence, vocab):
        return [Vocab.id_not_found if c not in vocab.word2id else vocab.word2id[c] for c in sentence]

    @classmethod
    def pad_batch_sequence(cls, sequences, pad_id, sos_id, eos_id, max_length = None):
        if max_length is None:
            max_length = max(map(lambda x: len(x), sequences))

        batch_sequences_in = pad_id * np.ones([len(sequences), max_length + 2], dtype=np.int32)
        batch_sequences_out = batch_sequences_in.copy()

        sequences_length = np.zeros(len(sequences), dtype=np.int32)
        for idx, sequence in enumerate(sequences):
            batch_sequences_in[idx, 0] = sos_id
            batch_sequences_in[idx, 1:len(sequence)+1] = np.asarray(sequence, dtype=np.int32)

            batch_sequences_out[idx, :len(sequence)] = np.asarray(sequence, dtype=np.int32)
            batch_sequences_out[idx, len(sequence)]  = eos_id

        return batch_sequences_in, batch_sequences_out, sequences_length

class ImageUtils:
    @classmethod
    def get_max_shape(cls, arrays):
        shapes = map(lambda x: list(x.shape), arrays)
        ndim = len(arrays[0].shape)
        max_shape = []
        for d in range(ndim):
            max_shape += [max(shapes, key=lambda x: x[d])[d]]

        return max_shape

    @classmethod
    def pad_batch_images(cls, images, max_shape=None):
        # 1. max shape
        if max_shape is None:
            max_shape = ImageUtils.get_max_shape(images)

        # 2. apply formating
        batch_images = 255 * np.ones([len(images)] + list(max_shape))
        for idx, img in enumerate(images):
            batch_images[idx, :img.shape[0], :img.shape[1]] = img

        return batch_images.astype(np.uint8)

    @classmethod
    def load(cls, img_fns, add_small_padding = True):
        imgs = [cv2.imread(img_fn,0) for img_fn in img_fns]

        if add_small_padding:
            imgs = [cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, (255,255,255)) for img in imgs]

        return imgs

    @classmethod
    def imgshow(cls, img):
        plt.imshow(img)
        plt.show()


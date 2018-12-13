import os
from src.utils import ImageUtils, TextUtils, Vocab

class DataLoader:
    count = 0

    def __init__(self, image_fodler, image_fn, label_fn, min_freq = 1, vocab=None):
        self.image_fns = [os.path.join(image_fodler, _fn.strip()) for _fn in open(image_fn, 'r').readlines()]
        self.labels = [_fn.strip() for _fn in open(label_fn, 'r').readlines()]

        assert len(self.image_fns) == len(self.labels)

        if vocab is None:
            dict_of_unique_words = TextUtils.get_dict_of_unique_words(self.labels, min_freq=min_freq)
            self.vocab = Vocab(dict_of_unique_words)
        else: self.vocab = vocab

    def gen_next(self, batch_size):
        _min = DataLoader.count
        _max = min(_min + batch_size, len(self.labels))

        _image_fns = self.image_fns[_min:_max]
        _image = ImageUtils.load(_image_fns)
        _label = self.labels[_min:_max]

        if _max > len(self.labels): DataLoader.count = 0
        return (_image, _label)

if __name__ == "__main__":
    pass
import argparse

from src.utils import Config, ImageUtils, TextUtils, Vocab
from src.data_loader.loader import DataLoader
from src.nn.model import Model

def print_welcolme():
    print ("Welcome")

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_folder", help="image_folder", type=str, default="./../data/images")
    parser.add_argument("--image_fn_train", help="image_filename_train", type=str, default="./../data/src-train.txt")
    parser.add_argument("--label_fn_train", help="label_filename_train", type=str, default="./../data/tgt-train.txt")
    parser.add_argument("--image_fn_valid", help="image_filename_valid", type=str, default="./../data/src-valid.txt")
    parser.add_argument("--label_fn_valid", help="label_filename_valid", type=str, default="./../data/tgt-valid.txt")

    parser.add_argument("--min_freq",help="min_freq",type=int,default=1)
    parser.add_argument("--max_ter",help="max_iter",type=int,default=150)

    parser.add_argument("--batch_size",help="batch_size",type=int,default=4)


    args = parser.parse_args()

    return args

def main():
    print_welcolme()
    args = get_parser()
    Config.set(args)

    # build data loader
    train_data_loader = DataLoader(Config.image_folder, Config.image_fn_train, Config.label_fn_train, Config.min_freq)
    valid_data_loader = DataLoader(Config.image_folder, Config.image_fn_valid, Config.label_fn_valid, 1,
                                   train_data_loader.vocab)

    # buid model
    model = Model()
    model.automatic_set(train_data_loader.vocab, args)
    model.build_multi_graph()

    # train
    while(1):
        # loading
        images, labels = train_data_loader.gen_next(args.get('batch_size',4))

        # preprocess
        normed_images  = ImageUtils.pad_batch_images(images=images)
        normed_labels_in, normed_labels_out, labels_length = TextUtils.pad_batch_sequence(sequences=labels,
                                                                                          pad_id=Vocab.pad,
                                                                                          sos_id=Vocab.sos,
                                                                                          eos_id=Vocab.eos)

        # train model
        loss, correct, sample_ids = model.run(images,normed_labels_in,normed_labels_out,labels_length,'training')












if __name__ == "__main__":
    main()
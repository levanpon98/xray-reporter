import os
import glob
import pickle
import argparse

import cv2
import time
import numpy as np
import tensorflow as tf
import pandas as pd

import config
from models.encoder import Encoder
from models.decoder import TranslayerDecoder
from utils import plot_attention
from extract_extract_key import get_keyvalue

def parser_args():
    parser = argparse.ArgumentParser(description='Inference model')
    parser.add_argument('-i', '--image-dir',
                        default='/home/minh/work/xraydata/data/images/images_normalized/',
                        metavar='image_path', type=str, help='path of image')
    parser.add_argument('-plot',
                        default=False, type=bool, help='Plot attention option')
    parser.add_argument('-test','--test-dir',
                        default='none',
                        metavar='image_path', type=str,help='path of testing image')

    return parser.parse_args()


def main():
    args = parser_args()

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    encoder = Encoder(config.embedding_dim)
    decoder = TranslayerDecoder(num_layers=config.num_layers,
                                embedding_dim=config.embedding_dim,
                                units=config.units,
                                num_heads=config.num_heads,
                                dff=config.dff,
                                vocab_size=config.vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    checkpoint_path = "../checkpoints/"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    print("Restore latest checkpoints ...")
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    def evaluation(img):
        text = []
        attention_plot = np.zeros((config.max_length, config.attention_features_shape))
        img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
        img = tf.expand_dims(img, axis=0)
        features = encoder(img)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        hidden = decoder.reset_state(batch_size=1)

        for i in range(config.max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            try:
                text.append(tokenizer.index_word[predicted_id])
                if tokenizer.index_word[predicted_id] == '<end>':
                    return text[:-1], attention_plot
            except:
                print('OOV')
                text.append('<oov>')
            

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(text), :]
        return text, attention_plot
    # for image_path in glob.glob(os.path.join(args.image_dir, '*.png')):
    #     print(image_path)
    #     image = cv2.imread(image_path)
    #     image = np.resize(image, (config.image_height, config.image_width, 3))

    #     if image.shape[2] == 1:
    #         image = np.dstack([image] * 3)
    #     else:
    #         image = image[:, :, :3]

    #     result, plot = evaluation(image)
    #     if args.plot:
    #         plot_attention(image, result, plot)
    #     print('Predict: ', ' '.join(result))
    
    labels = pd.read_csv("/home/minh/work/xraydata/data/data.csv")
    filenames = labels['filename'].tolist()[:50]
    labels = labels['findings'].tolist()[:50]
    preds = []
    start = time.time()
    for filename in filenames:
        image_path = os.path.join(args.image_dir, filename)
        image = cv2.imread(image_path)
        image = np.resize(image, (config.image_height, config.image_width, 3))

        if image.shape[2] == 1:
            image = np.dstack([image] * 3)
        else:
            image = image[:, :, :3]

        result, plot = evaluation(image)
        if args.plot:
            plot_attention(image, result, plot)
        print('Predict: ', ' '.join(result))
        preds.append(' '.join(result))
        dic = get_keyvalue(' '.join(result))
        print(dic)
        exit()
    print("Testing ended" ,time.time() - start)
    out = pd.DataFrame({
        'filename' : filenames,
        'predict' : preds,
        'label' : labels
    })
    out.to_csv('out2.csv')
if __name__ == '__main__':
    main()

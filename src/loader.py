import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from sklearn.model_selection import train_test_split

import config


def load_csv(data_root):
    contents = pd.read_csv(os.path.join(data_root, 'data.csv'))
    all_text = contents['findings'].map(lambda x: '<start> ' + x + ' <end>').astype(str).to_numpy()
    all_images = contents['filename'].map(lambda x: os.path.join(data_root, 'images/images_normalized', x)).astype(str).to_numpy()

    train_images, valid_images, train_texts, valid_texts = train_test_split(all_images, all_text, test_size=0.1,
                                                                            random_state=42)

    return train_images, valid_images, train_texts, valid_texts, all_text


def preprocess(image_path, caption):
    img_raw = tf.io.read_file(image_path)
    img_tensor = tf.image.decode_png(img_raw, channels=config.image_channels)
    img_tensor = tf.image.resize(img_tensor, (config.image_height, config.image_width))
    image_tensor = preprocess_input(img_tensor)

    return image_tensor, caption


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def load_ds(images, texts, tokenizer):
    train_seqs = tokenizer.texts_to_sequences(texts)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    max_length = calc_max_length(train_seqs)

    ds = tf.data.Dataset.from_tensor_slices((images, cap_vector))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(config.BATCH_SIZE)
    return ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE), max_length


def load_data(data_path):
    train_images, valid_images, train_texts, valid_texts, all_text = load_csv(data_path)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=config.top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$\t\n',
                                                      lower= True)

    tokenizer.fit_on_texts(all_text)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    train_ds, max_length_train = load_ds(train_images, train_texts, tokenizer)
    valid_ds, max_length_valid = load_ds(valid_images, valid_texts, tokenizer)

    return train_ds, valid_ds, len(train_images), len(valid_images), tokenizer

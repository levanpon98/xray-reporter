import time
import datetime

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from src import config
from src.loader import load_data
from src.models.encoder import Encoder
from src.models.decoder import Decoder


if __name__ == '__main__':

    train_ds, valid_ds, max_length_train, max_length_valid, tokenizer = load_data(config.data_path)

    encoder = Encoder(config.embedding_dim)
    decoder = Decoder(config.embedding_dim, config.units, config.vocab_size)
    optimizer = tf.keras.optimizers.Adam()
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    checkpoint_path = "./saved/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    current_time = datetime.datetime.now().strftime('%Y%m%d-H%M%S')
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary = tf.summary.create_file_writer(train_log_dir)
    test_summary = tf.summary.create_file_writer(test_log_dir)

    def loss_function(targets, logits):
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = crossentropy(targets, logits, sample_weight=mask)

        return loss

    def train_step(img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def valid_step(img_tensor, target):
        loss = 0

        hidden = decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        return loss, total_loss

    EPOCHS = 20

    for epoch in range(0, EPOCHS):

        # Training
        total_loss = 0
        pb_i = Progbar(max_length_train / config.BATCH_SIZE, stateful_metrics=['loss'])
        for (batch, (img_tensor, target)) in enumerate(train_ds):

            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss
            pb_i.add(batch, values=[('loss', t_loss)])

        with train_summary.as_default():
            tf.summary.scalar('train loss', total_loss, step=epoch)

        print('Total loss: {}'.format(total_loss / (max_length_train / config.BATCH_SIZE)))

        # Validation
        total_loss = 0
        pb_i = Progbar(max_length_train / config.BATCH_SIZE, stateful_metrics=['loss'])
        for (batch, (img_tensor, target)) in enumerate(valid_ds):
            batch_loss, t_loss = valid_ds(img_tensor, target)
            total_loss += t_loss
            pb_i.add(batch, values=[('loss', t_loss)])

        with test_summary.as_default():
            tf.summary.scalar('valid loss', total_loss, step=epoch)

        print('Total loss: {}'.format(total_loss / (max_length_train / config.BATCH_SIZE)))

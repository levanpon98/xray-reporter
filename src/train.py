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
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    current_time = datetime.datetime.now().strftime('%Y%m%d-H%M%S')
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary = tf.summary.create_file_writer(train_log_dir)
    test_summary = tf.summary.create_file_writer(test_log_dir)

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def train_step(img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)
            exit()
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


    EPOCHS = 20

    for epoch in range(0, EPOCHS):
        start = time.time()
        total_loss = 0

        pb_i = Progbar(max_length_train, stateful_metrics=['loss'])
        for (batch, (img_tensor, target)) in enumerate(train_ds):

            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss
            pb_i.add(config.BATCH_SIZE, values=[('loss', total_loss)])

        ckpt_manager.save()



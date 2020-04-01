# AUTOGENERATED! DO NOT EDIT! File to edit: 05_trainer.ipynb (unless otherwise specified).

__all__ = ['TrainTimeCallback', 'create_categorical_feature_column_with_hash_bucket',
           'create_categorical_feature_column_with_vocabulary_list', 'create_embedding',
           'create_linear_feature_columns', 'create_categorical_embeddings_feature_columns', 'create_feature_columns',
           'create_keras_model_sequential', 'train_and_evaluate_keras_model', 'train_and_evaluate_keras_model_small']

# Cell
from .constants import *
import datetime
import tensorflow as tf

class TrainTimeCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        logging.info('\nepoch train time: (hh:mm:ss.ms) {}'.format(
            datetime.datetime.now() - self.epoch_start_time))
        if not self.params is None:
            if 'steps' in self.params and self.params['steps']:
                epoch_milliseconds = (datetime.datetime.now(
                ) - self.epoch_start_time).total_seconds() * 1000
                logging.info(
                    '{} ms/step'.format(epoch_milliseconds / self.params['steps']))
                if BATCH_SIZE is not None:
                    logging.info('{} microseconds/example'.format(
                        1000 * epoch_milliseconds / self.params['steps'] / BATCH_SIZE))

# Cell
from .constants import *
import tensorflow as tf


def create_categorical_feature_column_with_hash_bucket(corpus_dict, key):
    corpus_size = len(corpus_dict[key])
    hash_bucket_size = min(corpus_size, 100000)
    categorical_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
        key,
        hash_bucket_size,
        dtype=tf.dtypes.string
    )
    logging.info('categorical column %s hash_bucket_size %d',
                 key, hash_bucket_size)
    return categorical_feature_column


def create_categorical_feature_column_with_vocabulary_list(corpus_dict, key):
    corpus_size = len(corpus_dict[key])
    categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key,
        list(corpus_dict[key].keys()),
        dtype=tf.dtypes.string,
        num_oov_buckets=corpus_size
    )
    logging.info(
        'categorical column with vocabular %s corpus_size %d', key, corpus_size)

    return categorical_feature_column

def create_embedding(vocabulary_size_dict, key, categorical_feature_column):
    vocabulary_size = vocabulary_size_dict[key]
    if vocabulary_size < 10:
        logging.info(
            'categorical column %s vocabulary_size %d - creating indicator column', key, vocabulary_size)
        return tf.feature_column.indicator_column(categorical_feature_column)

    embedding_dimension = int(min(50, math.floor(6 * vocabulary_size**0.25)))
    embedding_feature_column = tf.feature_column.embedding_column(
        categorical_feature_column,
        embedding_dimension)
    return embedding_feature_column

def create_linear_feature_columns():
    return list(tf.feature_column.numeric_column(field.name, dtype=tf.dtypes.float32) for field in CSV_SCHEMA if field.field_type == 'INTEGER' and field.name != 'label')

def create_categorical_embeddings_feature_columns(corpus_dict, vocabulary_size_dict, embeddings_mode: EMBEDDINGS_MODE_TYPE):
    if embeddings_mode == EMBEDDINGS_MODE_TYPE.none:
        return []
    elif embeddings_mode == EMBEDDINGS_MODE_TYPE.hashbucket:
        return list(create_embedding(
            vocabulary_size_dict,
            key,
            create_categorical_feature_column_with_hash_bucket(corpus_dict, key))
            for key, _ in corpus_dict.items())
    elif embeddings_mode == EMBEDDINGS_MODE_TYPE.vocabular:
        return list(create_embedding(
            vocabulary_size_dict,
            key,
            create_categorical_feature_column_with_vocabulary_list(corpus_dict, key))
            for key, _ in corpus_dict.items())
    else:
        raise ValueError('invalid embedding_mode: {}'.format(embedding_mode))


# Cell
import criteo_nbdev.data_reader

def create_feature_columns(embedding_mode: EMBEDDINGS_MODE_TYPE):
    corpus_dict = criteo_nbdev.data_reader.get_corpus_dict()
    vocabulary_size_dict = criteo_nbdev.data_reader.get_vocabulary_size_dict()
    feature_columns = []
    feature_columns.extend(create_linear_feature_columns())
    feature_columns.extend(
        create_categorical_embeddings_feature_columns(corpus_dict, vocabulary_size_dict, embedding_mode))
    return feature_columns

# Cell

from .constants import *
import tensorflow as tf

def create_keras_model_sequential():
    feature_columns = create_feature_columns(EMBEDDINGS_MODE_TYPE.hashbucket)

    feature_layer = tf.keras.layers.DenseFeatures(
        feature_columns, name="feature_layer")
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
    BatchNormalization = tf.keras.layers.BatchNormalization
    model = tf.keras.Sequential(
        [
            feature_layer,
            Dropout(0.3),
            Dense(598, activation=tf.nn.relu),
            Dense(598, activation=tf.nn.relu),
            Dense(598, activation=tf.nn.relu),
            Dense(1, activation=tf.nn.sigmoid)
        ])

    logging.info('compiling sequential keras model')
    # Compile Keras model
    model.compile(
        # cannot use Adagrad with mirroredstartegy https://github.com/tensorflow/tensorflow/issues/19551
        # optimizer=tf.optimizers.Adagrad(learning_rate=0.05),
        optimizer=tf.optimizers.SGD(learning_rate=0.05),
        # optimizer=tf.optimizers.Adam(learning_rate=0.0005),
        # optimizer=tf.optimizers.Adam(),
        #optimizer=tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.1),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])
    return model

# Cell
from .constants import *
from gcp_runner.ai_platform_constants import *
import criteo_nbdev.data_reader
import nbdev.imports
import tensorflow as tf
import logging
import math
import os

def train_and_evaluate_keras_model(
    model,
    model_dir,
    epochs,
    dataset_source: DATASET_SOURCE_TYPE,
    dataset_size: DATASET_SIZE_TYPE,
    embeddings_mode: EMBEDDINGS_MODE_TYPE,
    distribution_strategy: DistributionStrategyType):

    log_dir = os.path.join(model_dir, "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        embeddings_freq=1,
        profile_batch=min(epochs, 2))

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    # crashing https://github.com/tensorflow/tensorflow/issues/27688
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    callbacks = []
    train_time_callback = TrainTimeCallback()

    if DistributionStrategyType == DistributionStrategyType.TPU_STRATEGY:
        # epoch and accuracy constants are not supported when training on TPU.
        checkpoints_file_path = checkpoints_dir + "/epochs_tpu.hdf5"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoints_file_path, verbose=1, mode='max')
        callbacks = [tensorboard_callback,
                     checkpoint_callback, train_time_callback]
    else:
        if embeddings_mode == EMBEDDINGS_MODE_TYPE.manual or distribution_strategy == DistributionStrategyType.MULTI_WORKER_MIRRORED_STRATEGY:
            # accuracy fails for adagrad
            # for some reason accuracy is not available for EMBEDDINGS_MODE_TYPE.manual
            # for some reason accuracy is not available for MultiWorkerMirroredStrategy
            checkpoints_file_path = checkpoints_dir + \
                "/epochs:{epoch:03d}.hdf5"
        else:
            checkpoints_file_path = checkpoints_dir + \
                "/epochs:{epoch:03d}-accuracy:{accuracy:.3f}.hdf5"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoints_file_path, verbose=1, mode='max')
        callbacks = [tensorboard_callback, checkpoint_callback, train_time_callback]

    verbosity = 1 if nbdev.imports.in_ipython() else 2
    logging.info('training keras model')
    training_ds = criteo_nbdev.data_reader.get_dataset(dataset_source, dataset_size, DATASET_TYPE.training, embeddings_mode).repeat(epochs)
    eval_ds = criteo_nbdev.data_reader.get_dataset(dataset_source, dataset_size, DATASET_TYPE.validation, embeddings_mode).repeat(epochs)

    # steps_per_epoch and validation_steps are required for MultiWorkerMirroredStrategy
    model.fit(
        training_ds,
        epochs=epochs,
        verbose=verbosity,
        callbacks=callbacks,
        steps_per_epoch=criteo_nbdev.data_reader.get_steps_per_epoch(dataset_size, DATASET_TYPE.training),
        validation_data=eval_ds,
        validation_steps=criteo_nbdev.data_reader.get_steps_per_epoch(dataset_size, DATASET_TYPE.validation))

    logging.info("done training keras model, evaluating model")
    loss, accuracy = model.evaluate(
        eval_ds,
        verbose=verbosity,
        steps=criteo_nbdev.data_reader.get_steps_per_epoch(dataset_size, DATASET_TYPE.validation),
        callbacks=[tensorboard_callback])
    logging.info("Eval - Loss: {}, Accuracy: {}".format(loss, accuracy))
    logging.info(model.summary())
    logging.info("done evaluating keras model")
    return {'accuracy': accuracy, 'loss': loss}


# Cell

import gcp_runner.core
gcp_runner.core.export_and_reload_all(silent=True)

# def train_keras_sequential(strategy, model_dir):
#     return train_and_evaluate_keras_model(create_keras_model_sequential(), model_dir)

# train_keras_sequential(None, './models/model1')

def train_and_evaluate_keras_model_small(distribution_strategy=None, job_dir=None, int_arg=None, **kwargs):
    print('distribution_strategy:')
    print(distribution_strategy)
    print('job_dir:')
    print(job_dir)
    print('int_arg:')
    print(int_arg)
    print(type(int_arg))
    print('kwargs:')
    print(kwargs)
#     print('args:')7
#     print(args)
    #train_and_evaluate_keras_model(create_keras_model_sequential(), './models/model1', 2, DATASET_SOURCE_TYPE.bq, DATASET_SIZE_TYPE.full, EMBEDDINGS_MODE_TYPE.hashbucket, None)


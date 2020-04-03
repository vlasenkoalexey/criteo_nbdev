# AUTOGENERATED! DO NOT EDIT! File to edit: 05_trainer.ipynb (unless otherwise specified).

__all__ = ['TrainTimeCallback', 'PlotLossesCallback', 'create_categorical_feature_column_with_hash_bucket',
           'create_categorical_feature_column_with_vocabulary_list', 'create_embedding',
           'create_linear_feature_columns', 'create_categorical_embeddings_feature_columns', 'create_feature_columns',
           'create_keras_model_sequential', 'train_and_evaluate_keras_model', 'train_and_evaluate_keras',
           'train_and_evaluate_estimator', 'train_and_evaluate_keras_model_small',
           'train_and_evaluate_estimator_model_small']

# Cell

# just updated now

from .constants import *
from gcp_runner.ai_platform_constants import *

# Cell

import datetime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

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



class PlotLossesCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

# Cell
import tensorflow as tf

def create_categorical_feature_column_with_hash_bucket(vocabulary_size_dict, key):
    corpus_size = vocabulary_size_dict[key]
    hash_bucket_size = min(corpus_size, 100000)
    categorical_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
        key,
        hash_bucket_size,
        dtype=tf.dtypes.string
    )
    logging.info('categorical column %s hash_bucket_size %d',
                 key, hash_bucket_size)
    return categorical_feature_column

def create_categorical_feature_column_with_vocabulary_list(corpus_dict, vocabulary_size_dict, key):
    corpus_size = vocabulary_size_dict[key]
    categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key,
        list(corpus_dict[key].keys()),
        dtype=tf.dtypes.string,
        num_oov_buckets=corpus_size - len(corpus_dict[key])
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

def create_categorical_embeddings_feature_columns(vocabulary_size_dict, embeddings_mode: EMBEDDINGS_MODE_TYPE):
    if embeddings_mode == EMBEDDINGS_MODE_TYPE.none:
        return []
    elif embeddings_mode == EMBEDDINGS_MODE_TYPE.hashbucket:
        return list(create_embedding(
            vocabulary_size_dict,
            key,
            create_categorical_feature_column_with_hash_bucket(vocabulary_size_dict, key))
            for key, _ in vocabulary_size_dict.items())
    elif embeddings_mode == EMBEDDINGS_MODE_TYPE.vocabular:
        logging.info('loading corpus dictionary')
        corpus_dict = criteo_nbdev.data_reader.get_corpus_dict()
        return list(create_embedding(
            vocabulary_size_dict,
            key,
            create_categorical_feature_column_with_vocabulary_list(corpus_dict, vocabulary_size_dict, key))
            for key, _ in corpus_dict.items())
    else:
        raise ValueError('invalid embedding_mode: {}'.format(embedding_mode))

def create_feature_columns(embedding_mode: EMBEDDINGS_MODE_TYPE):
    logging.info('loading vocabulary size dictionary')
    vocabulary_size_dict = criteo_nbdev.data_reader.get_vocabulary_size_dict()
    feature_columns = []
    feature_columns.extend(create_linear_feature_columns())
    feature_columns.extend(
        create_categorical_embeddings_feature_columns(vocabulary_size_dict, embedding_mode))
    return feature_columns

# Cell

import criteo_nbdev.data_reader
import nbdev.imports
import tensorflow as tf
import logging
import math
import os

def create_keras_model_sequential(embeddings_mode: EMBEDDINGS_MODE_TYPE):
    feature_columns = create_feature_columns(embeddings_mode)

    feature_layer = tf.keras.layers.DenseFeatures(
        feature_columns, name="feature_layer")
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
    BatchNormalization = tf.keras.layers.BatchNormalization
    model = tf.keras.Sequential(
        [
            feature_layer,
            Dense(598, activation=tf.nn.relu),
            Dense(598, activation=tf.nn.relu),
            Dense(598, activation=tf.nn.relu),
            Dense(1, activation=tf.nn.sigmoid)
        ])

    logging.info('compiling sequential keras model')
    # Compile Keras model
    model.compile(
        optimizer=tf.optimizers.SGD(learning_rate=0.05),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])
    return model

def train_and_evaluate_keras_model(
    model,
    model_dir,
    epochs = 3,
    dataset_source: DATASET_SOURCE_TYPE = DATASET_SOURCE_TYPE.bq,
    dataset_size: DATASET_SIZE_TYPE = DATASET_SIZE_TYPE.tiny,
    embeddings_mode: EMBEDDINGS_MODE_TYPE = EMBEDDINGS_MODE_TYPE.hashbucket,
    distribution_strategy: DistributionStrategyType = None):

    log_dir = os.path.join(model_dir, "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        embeddings_freq=1)

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
    if nbdev.imports.in_ipython():
        callbacks.append(PlotLossesCallback())

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

def train_and_evaluate_keras(
    model_dir,
    epochs = 3,
    dataset_source: DATASET_SOURCE_TYPE = DATASET_SOURCE_TYPE.bq,
    dataset_size: DATASET_SIZE_TYPE = DATASET_SIZE_TYPE.tiny,
    embeddings_mode: EMBEDDINGS_MODE_TYPE = EMBEDDINGS_MODE_TYPE.hashbucket,
    distribution_strategy: DistributionStrategyType = None):

    model = create_keras_model_sequential(embeddings_mode)
    return train_and_evaluate_keras_model(
        model,
        model_dir,
        epochs=epochs,
        dataset_source=dataset_source,
        dataset_size=dataset_size,
        embeddings_mode=embeddings_mode,
        distribution_strategy=distribution_strategy)


# Cell

def train_and_evaluate_estimator(
    model_dir,
    epochs = 3,
    dataset_source: DATASET_SOURCE_TYPE = DATASET_SOURCE_TYPE.bq,
    dataset_size: DATASET_SIZE_TYPE = DATASET_SIZE_TYPE.tiny,
    embeddings_mode: EMBEDDINGS_MODE_TYPE = EMBEDDINGS_MODE_TYPE.hashbucket,
    distribution_strategy: DistributionStrategyType = None):

    print(dataset_size)
    logging.info('training for {} steps'.format(criteo_nbdev.data_reader.get_steps_per_epoch(dataset_size, DATASET_TYPE.training)))
    config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        eval_distribute=distribution_strategy)

    feature_columns = create_feature_columns(embeddings_mode)
    estimator = tf.estimator.DNNClassifier(
        optimizer=tf.optimizers.SGD(learning_rate=0.05),
        feature_columns=feature_columns,
        hidden_units=[598, 598, 598],
        model_dir=model_dir,
        config=config,
        n_classes=2)
    logging.info('training and evaluating estimator model')
    training_ds = criteo_nbdev.data_reader.get_dataset(dataset_source, dataset_size, DATASET_TYPE.training, embeddings_mode).repeat(epochs)
    eval_ds = criteo_nbdev.data_reader.get_dataset(dataset_source, dataset_size, DATASET_TYPE.validation, embeddings_mode).repeat(epochs)

    # Need to specify both max_steps and epochs. Each worker will go through epoch separately.
    # see https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate?version=stable
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=lambda: criteo_nbdev.data_reader.get_dataset(dataset_source, dataset_size, DATASET_TYPE.training, embeddings_mode).repeat(epochs),
            max_steps=criteo_nbdev.data_reader.get_steps_per_epoch(dataset_size, DATASET_TYPE.training) * epochs),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=lambda: criteo_nbdev.data_reader.get_dataset(dataset_source, dataset_size, DATASET_TYPE.validation, embeddings_mode).repeat(epochs)))
    logging.info('done evaluating estimator model')

# Cell

import gcp_runner.core
gcp_runner.core.export_and_reload_all(silent=True)

# def train_keras_sequential(strategy, model_dir):
#     return train_and_evaluate_keras_model(create_keras_model_sequential(), model_dir)

# train_keras_sequential(None, './models/model1')

def train_and_evaluate_keras_model_small(distribution_strategy=None, job_dir=None, **kwargs):
    print('distribution_strategy:')
    print(distribution_strategy)
    print('job_dir:')
    print(job_dir)
    print('kwargs:')
    print(kwargs)
    train_and_evaluate_keras_model(create_keras_model_sequential(EMBEDDINGS_MODE_TYPE.hashbucket), job_dir, 2, DATASET_SOURCE_TYPE.bq, DATASET_SIZE_TYPE.tiny, EMBEDDINGS_MODE_TYPE.hashbucket, None)

def train_and_evaluate_estimator_model_small(distribution_strategy=None, job_dir=None, **kwargs):
    print('distribution_strategy:')
    print(distribution_strategy)
    print('job_dir:')
    print(job_dir)
    print('kwargs:')
    print(kwargs)
    train_estimator(job_dir, 2, DATASET_SOURCE_TYPE.bq, DATASET_SIZE_TYPE.tiny, EMBEDDINGS_MODE_TYPE.hashbucket, distribution_strategy)
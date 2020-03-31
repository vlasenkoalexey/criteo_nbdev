# AUTOGENERATED! DO NOT EDIT! File to edit: 01_constants.ipynb (unless otherwise specified).

__all__ = ['PROJECT_ID', 'LOCATION', 'SERVICE_ACCOUNT_KEY_FILE_NAME', 'GCS_BUCKET', 'GCS_FOLDER', 'PRIVATE_GCS_BUCKET',
           'URL', 'DATASET_ID', 'TABLE_ID', 'BATCH_SIZE', 'DATASET_SIZE_TYPE', 'DATASET_SIZE', 'DATASET_SOURCE_TYPE',
           'DATASET_SOURCE', 'DATASET_TYPE', 'EMBEDDINGS_MODE_TYPE', 'CSV_SCHEMA', 'FULL_TRAINING_DATASET_SIZE',
           'FULL_VALIDATION_DATASET_SIZE', 'SMALL_TRAINING_DATASET_SIZE', 'SMALL_VALIDATION_DATASET_SIZE']

# Cell
from enum import Enum
from google.cloud import bigquery

PROJECT_ID = "alekseyv-scalableai-dev" #@param {type:"string"}
LOCATION = 'us'
SERVICE_ACCOUNT_KEY_FILE_NAME = 'service_account_key.json'

GCS_BUCKET = 'gs://alekseyv-scalableai-dev-public-bucket/'
GCS_FOLDER = GCS_BUCKET + 'criteo_kaggle/'
PRIVATE_GCS_BUCKET = 'gs://alekseyv-scalableai-dev-private-bucket'

URL = GCS_BUCKET + 'criteo_kaggle_decompressed/train.txt'
DATASET_ID = 'criteo_kaggle_2'
TABLE_ID = 'days'

BATCH_SIZE = 512

DATASET_SIZE_TYPE = Enum('DATASET_SIZE_TYPE', 'full small')
DATASET_SIZE = DATASET_SIZE_TYPE.small

DATASET_SOURCE_TYPE = Enum('DATASET_SOURCE_TYPE', 'bq gcs')
DATASET_SOURCE = DATASET_SOURCE_TYPE.bq

DATASET_TYPE = Enum('DATASET_TYPE', 'training validation')

EMBEDDINGS_MODE_TYPE = Enum('EMBEDDINGS_MODE_TYPE',
                            'none manual hashbucket vocabular')

CSV_SCHEMA = [
      bigquery.SchemaField("label", "INTEGER", mode='REQUIRED'),
      bigquery.SchemaField("int1", "INTEGER"),
      bigquery.SchemaField("int2", "INTEGER"),
      bigquery.SchemaField("int3", "INTEGER"),
      bigquery.SchemaField("int4", "INTEGER"),
      bigquery.SchemaField("int5", "INTEGER"),
      bigquery.SchemaField("int6", "INTEGER"),
      bigquery.SchemaField("int7", "INTEGER"),
      bigquery.SchemaField("int8", "INTEGER"),
      bigquery.SchemaField("int9", "INTEGER"),
      bigquery.SchemaField("int10", "INTEGER"),
      bigquery.SchemaField("int11", "INTEGER"),
      bigquery.SchemaField("int12", "INTEGER"),
      bigquery.SchemaField("int13", "INTEGER"),
      bigquery.SchemaField("cat1", "STRING"),
      bigquery.SchemaField("cat2", "STRING"),
      bigquery.SchemaField("cat3", "STRING"),
      bigquery.SchemaField("cat4", "STRING"),
      bigquery.SchemaField("cat5", "STRING"),
      bigquery.SchemaField("cat6", "STRING"),
      bigquery.SchemaField("cat7", "STRING"),
      bigquery.SchemaField("cat8", "STRING"),
      bigquery.SchemaField("cat9", "STRING"),
      bigquery.SchemaField("cat10", "STRING"),
      bigquery.SchemaField("cat11", "STRING"),
      bigquery.SchemaField("cat12", "STRING"),
      bigquery.SchemaField("cat13", "STRING"),
      bigquery.SchemaField("cat14", "STRING"),
      bigquery.SchemaField("cat15", "STRING"),
      bigquery.SchemaField("cat16", "STRING"),
      bigquery.SchemaField("cat17", "STRING"),
      bigquery.SchemaField("cat18", "STRING"),
      bigquery.SchemaField("cat19", "STRING"),
      bigquery.SchemaField("cat20", "STRING"),
      bigquery.SchemaField("cat21", "STRING"),
      bigquery.SchemaField("cat22", "STRING"),
      bigquery.SchemaField("cat23", "STRING"),
      bigquery.SchemaField("cat24", "STRING"),
      bigquery.SchemaField("cat25", "STRING"),
      bigquery.SchemaField("cat26", "STRING")
  ]

FULL_TRAINING_DATASET_SIZE = 42840617
FULL_VALIDATION_DATASET_SIZE = 3000000
SMALL_TRAINING_DATASET_SIZE = 4500000
SMALL_VALIDATION_DATASET_SIZE = 500000
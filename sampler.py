#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

def main(spark):

    partitions = 1000

    train_data = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_train.parquet')
    validation_data = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_validation.parquet')
    test_data = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_test.parquet')
    
    val_test_data = test_data.union(validation_data)
    train_optional_user = train_data.join(val_test_data, "user_id", "left_anti")
    train_data = train_data.join(train_optional_user, "user_id", "left_anti")
        
    uid_indexer = StringIndexer(inputCol="user_id", outputCol="user_num", handleInvalid="skip")
    tid_indexer = StringIndexer(inputCol="track_id", outputCol="track_num", handleInvalid="skip")
    model_uid = uid_indexer.fit(train_data)
    model_tid = tid_indexer.fit(train_data)

    uid_train_index = model_uid.transform(train_data)
    combo_train_index = model_tid.transform(uid_train_index)
    
    uid_val_index = model_uid.transform(validation_data)
    combo_val_index = model_tid.transform(uid_val_index)
    
    uid_test_index = model_uid.transform(test_data)
    combo_test_index = model_tid.transform(uid_test_index)
    
    model_uid.save('anshul_project/model_uid')
    model_tid.save('anshul_project/model_tid')

    combo_train_index = combo_train_index.repartition(partitions, "user_id")
    combo_val_index = combo_val_index.repartition(partitions, "user_id")
    combo_test_index = combo_test_index.repartition(partitions, "user_id")

    combo_train_index = combo_train_index.select(["user_num","count","track_num"])
    combo_train_index.write.parquet(path='anshul_project/train_index.parquet', mode='overwrite')
    combo_train_index.unpersist()

    combo_val_index = combo_val_index.select(["user_num","count","track_num"])
    combo_val_index.write.parquet(path='anshul_project/val_index.parquet', mode='overwrite')
    combo_val_index.unpersist()
    
    combo_test_index = combo_test_index.select(["user_num","count","track_num"])
    combo_test_index.write.parquet(path='anshul_project/test_index.parquet', mode='overwrite')
    combo_test_index.unpersist()


# Only enter this block if we're in main
if __name__ == "__main__":
    spark = SparkSession.builder.appName('sampler').getOrCreate()
    main(spark)

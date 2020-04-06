#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.appName('test').getOrCreate()
    als_model = ALSModel.load('anshul_project/als_sampling')
   
    test_data = spark.read.parquet('anshul_project/test_index.parquet')

    als_predictions = als_model.transform(test_data)

    reg_evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
    rmse = reg_evaluator.evaluate(als_predictions)

    print("Test rmse " + str(rmse))


# Only enter this block if we're in main
if __name__ == "__main__":
    main()

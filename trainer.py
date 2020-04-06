#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def main(spark):    

    train_data = spark.read.parquet('anshul_project/train_index.parquet')
    val_data = spark.read.parquet('anshul_project/val_index.parquet')
    train_data = train_data.cache()
    val_data = val_data.cache() 
    
    train_data = train_data.sample(withReplacement=False, fraction=0.1, seed=1)

    ranks = [4, 8, 16]
    regs = [0, 0.1, 1, 10]
    alphas = [0.1, 0.5, 1.0]
    # ranks = [4]
    # regs = [1]
    # alphas = [0.5]

    best_rmse = sys.maxsize
    best_model = None
    for rank in ranks:
        for reg in regs:
            for alpha in alphas:
                als = ALS(maxIter=3, rank=rank, regParam=reg, alpha=alpha, userCol="user_num", itemCol="track_num", ratingCol="count", implicitPrefs=True, coldStartStrategy="drop")
                als_model = als.fit(train_data)
            
                predictions = als_model.transform(val_data)
                reg_evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
                rmse = reg_evaluator.evaluate(predictions)
                
                if rmse < best_rmse:
                    best_model = als_model
                    best_rmse = rmse
                    print('New best model')
                    print('Rank: {}, Reg: {}, Alpha: {}'.format(rank, reg, alpha))
                    stats = [rank, reg, alpha, rmse]
                
    best_model.save('anshul_project/als_sampling')
    print('Best model: Rank: {}, Reg: {}, Alpha: {}, RMSE: {}'.format(*stats))


# Only enter this block if we're in main
if __name__ == "__main__":
    spark = SparkSession.builder.appName('als_train').getOrCreate()
    main(spark)

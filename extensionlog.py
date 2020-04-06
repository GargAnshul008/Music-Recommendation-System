import sys

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import log



def main(spark):
    train_data = spark.read.parquet('anshul_project/train_index.parquet')
    val_data = spark.read.parquet('anshul_project/val_index.parquet')

    # train_data.createOrReplaceTempView('train_data')

    train_data_log = train_data.withColumn("logcount", log(train_data["count"]))
    val_data_log = val_data.withColumn("logcount", log(val_data["count"]))
    uid_indexer = StringIndexer(inputCol="user_id", outputCol="user_num", handleInvalid ="skip")
    tid_indexer = StringIndexer(inputCol="track_id", outputCol="track_num", handleInvalid ="skip")

    ranks =[4]
    regs = [1]
    alphas = [0.5]
    
    best_rmse = None
    best_rank = None
    best_alpha = None
    best_reg = None

    for rank in ranks :
        for alpha in alphas :
            for reg in regs :

                als = ALS(maxIter = 3 , regParam= reg, userCol= "user_num" , itemCol= "track_num" , ratingCol ="logcount" , implicitPrefs=True , coldStartStrategy="drop" , alpha= a , rank = r)

                pipeline = Pipeline(stages=[uid_indexer, tid_indexer, als])
                
                als_model = pipeline.fit(train_data_log)
                predictions = als_model.transform(val_data_log)
                
                evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
                
                rmse = evaluator.evaluate(predictions)

                if best_rmse is None or best_rmse > rmse :
                   best_rmse = rmse
                   best_rank = rank
                   best_alpha = alpha
                   best_reg = reg

    print('The best hyper parameters: Rank: {}, Reg: {}, Alpha: {}, RMSE: {}'.format(best_rank,best_alpha,best_reg,best_rmse))


    als_model.save('anshul_project/log_model')
    pass

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('log_extension').getOrCreate()
    main(spark)

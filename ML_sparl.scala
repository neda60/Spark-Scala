import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostClassificationModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.Model
import spark.implicits._
import org.apache.spark.sql._
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.ml.{ Pipeline, PipelineModel, PipelineStage }
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, IndexToString, VectorAssembler}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Model, Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

import org.apache.spark.ml.param.shared
val sqlContext = new org.apache.spark.sql.SQLContext(sc)

val df_data = sqlContext.parquetFile("s3://project/*").
na.fill(0, Seq("device_type_id", "video_length", "exch_id", "user_type", "advertiser_id", "ip")).
na.fill("0", Seq("player", "city","region"))


df_data.registerTempTable("t_table")

val df = sqlContext.sql("SELECT (period % 10000)/100 day, period/100 hour,* FROM t_table where uid_type is not Null and domain_bundle is not Null and user_agent_hash is not Null and domain_bundle like '%.%' and device_type_id not in(4,5) and (lbl=0 or (period between 2019020100 and 2019023100 and lbl=1))")// 


val colsToRemove = Seq("bid_request_id" , "spot_id", "period") //"apple_ifa", "google_ifa", "amazon_fire_aid", "bid_request_id","campaign_id","flight_id","creative_id")
// dropping unwanted columns
val df_data = df.select(df.columns .filter(colName => !colsToRemove.contains(colName)) .map(colName => new Column(colName)): _*)

df_data.groupBy("lbl").count.show

// val df_data=df
val splits = df_data.randomSplit(Array(0.7, 0.3), seed = 24L)//, 0.02
val training_data = splits(0).cache()
val test_data = splits(1)

// to avoid regenrating random data
training_data.write.save("")
test_data.write.save("")

val stringIndexer_label = new StringIndexer().setInputCol("lbl").setOutputCol("label").fit(df_data)

val playerIndexer = new StringIndexer().setInputCol("player").setOutputCol("player_index").setHandleInvalid("keep")

val domain_bundleIndexer = new StringIndexer().setInputCol("domain_bundle").setOutputCol("domain_bundle_index").setHandleInvalid("keep")

val cityIndexer = new StringIndexer().setInputCol("city").setOutputCol("city_index").setHandleInvalid("keep")

val regionIndexer = new StringIndexer().setInputCol("region").setOutputCol("region_index").setHandleInvalid("keep")
//
val vectorAssembler_features = new VectorAssembler().
setInputCols(Array("player_index", "city_index","region_index","domain_bundle_index", "uid_type","advertiser_id" ,"day", "hour", "video_duration", "exchange_id", "device_type_id", "ip", "user_agent_hash")).
setOutputCol("features")

val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(101).setMaxDepth(6).setMaxBins(100000000)

val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(stringIndexer_label.labels)

val pipelineRF = new Pipeline().setStages(Array(stringIndexer_label,domain_bundleIndexer, playerIndexer, cityIndexer, regionIndexer,vectorAssembler_features, rf))
//

val model_rf = pipelineRF.fit(training_data)
val predictions = model_rf.transform(test_data)
val evaluatorRF = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluatorRF.evaluate(predictions)

println(f"Accuracy = $accuracy%.2f")

// PROBLEM HERE EVEN WITH SMALL SET FALLS INTO AN INFINITIVE LOOP
val paramGridRF = new ParamGridBuilder().
// addGrid(rf.maxBins, Array(100,200)).
addGrid(rf.maxDepth, Array(2,4,10)).
addGrid(rf.numTrees, Array( 11, 51, 101)).
addGrid(rf.impurity, Array("entropy", "gini")).
build()

// val paramGridRF = new ParamGridBuilder().
// // addGrid(rf.maxBins, Array(100,200)).
// addGrid(rf.maxDepth, Array(10)).
// addGrid(rf.numTrees, Array( 11)).
// addGrid(rf.impurity, Array("entropy")).
// build()

val evaluatorRF= new BinaryClassificationEvaluator().
setLabelCol("label").
setRawPredictionCol("prediction")

val crossvalRF = new CrossValidator().
setEstimator(pipelineRF).
setEvaluator(evaluatorRF).
setEstimatorParamMaps(paramGridRF).
setNumFolds(3)

val pipelineModelRF = crossvalRF.fit(training_data)
// //Feature Importance 35:22
val bestModelRF = pipelineModelRF.bestModel.asInstanceOf[PipelineModel]

val size = bestModelRF.stages.size-1

val featureImportances = bestModelRF.stages(size).asInstanceOf[RandomForestClassificationModel].featureImportances.toArray


// Best parameters
//val bestEstimatorParamMapRF = pipelineModelRF.getEstimatorParamMaps.zip(pipelineModelRF.avgMetrics).maxBy(_._2)._1
val bestEstimatorParamMapRF = pipelineModelRF.getEstimatorParamMaps.zip(pipelineModelRF.avgMetrics).maxBy(_._2)._1
println($"Best params:\n $bestEstimatorParamMapRF")

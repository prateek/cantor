package com.cloudera.cantor

import akka.actor.ScalaActorRef
import com.cloudera.cantor.pipeline.FeatureSelector
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vector


case class Passenger(id: Int, survived: Double
                     , Pclass: Int, Name: String
                     , Sex: String, Age: Int
                     , SibSp:Int, Parch: Int
                     , Fare: Double, Embarked: String)

object FeatureSelectorExample {

  val SMART_CSV_SPLIT_REGEX = ",(?=([^\"]*\"[^\"]*\")*[^\"]*$)";
  def safeInt(x: String)={
    try {
      x.toInt
    } catch {
      case e: Exception => 0
    }
  }

  def loadData(sc: SparkContext): DataFrame ={
    val sqlContext = new SQLContext(sc)
    // used to implicitly convert RDD -> DF
    import sqlContext.implicits._
    // Load and parse the data file
    val data = sc.textFile("file:/Users/prungta/trash/cantor/src/main/resources/sample-data/titanic-train.csv")
    val parsedData = data.map(line => {
      val p = line.split(SMART_CSV_SPLIT_REGEX,-1)
      //for((x,i) <- p.view.zipWithIndex) println("String #" + i + " is " + x)
      Passenger(p(0).toInt, p(1).toDouble, p(2).toInt, p(3).trim, p(4).trim, safeInt(p(5)),
        p(6).toInt, p(7).toInt, p(9).toDouble, p(11))
    })
    parsedData.toDF
  }

  def main(args : Array[String]) {
    val conf = new SparkConf()
      .setAppName("Feature Selector Demo")
      .setMaster("local")
    val sc = new SparkContext(conf)

    val dataset = loadData(sc)
    val pClassIndexer = new StringIndexer()
      .setInputCol("Pclass")
      .setOutputCol("PclassIndex")
    val SexIndexer = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("SexIndex")
    val EmbarkedIndexer = new StringIndexer()
      .setInputCol("Embarked")
      .setOutputCol("EmbarkedIndex")

    def featureCombinations(combineList: Array[String], k: Int): Array[Array[String]] = {
      combineList.combinations(k).toArray.map(_.toArray) // ++ includeList)
    }

    val featureColumns = Array("PclassIndex"
      , "SexIndex", "Age", "SibSp", "Parch"
      , "Fare", "EmbarkedIndex")

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setMaxIter(10)

    val pipeline = new Pipeline()
      .setStages(Array(pClassIndexer, SexIndexer, EmbarkedIndexer, assembler, lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(assembler.inputCols, featureCombinations(featureColumns, featureColumns.length-1))
      .build()

    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = crossval.fit(dataset.withColumnRenamed("survived", "label"))
    // TODO: print best params
    // println("Best params ")
    sc.stop()
  }
}

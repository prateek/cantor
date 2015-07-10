package com.cloudera.cantor

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors


// Based on the examples from
// https://spark.apache.org/docs/1.0.1/mllib-decision-tree.html#examples
// https://spark.apache.org/docs/latest/ml-guide.html#example-model-selection-via-cross-validation

object FeatureSelectorExample {

  def main(args : Array[String]) {
    val conf = new SparkConf()
      .setAppName("Feature Selector Demo")
      .setMaster("local")
    val sc = new SparkContext(conf)

    // Load and parse the data file
    val data = sc.textFile("file:/Users/prungta/trash/cantor/src/main/resources/sample-data/sample_tree_data.csv")
    val parsedData = data.map { line =>
      val parts = line.split(',').map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }

    // Split the data into training and test sets (30% held out for testing)
    val splits = parsedData.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification tree model:\n" + model.toDebugString)

    sc.stop()
  }

}

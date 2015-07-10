package com.cloudera.cantor.pipeline

import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.{sql, SparkConf, SparkContext}
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import org.scalatest.{BeforeAndAfter, FunSuite}
import org.scalatest.matchers.ShouldMatchers

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.specs2.mutable.BeforeAfter

@RunWith(classOf[JUnitRunner])
class FeatureSelectorTest
  extends FunSuite with ShouldMatchers with BeforeAndAfter
{

  var conf:SparkConf  = _
  var sc:SQLContext = _
  var df:DataFrame = _

  before{
    conf = new SparkConf()
                .setMaster("local")
                .setAppName(getClass.getName)
    sc = new SQLContext(new SparkContext(conf))
    df = sc.createDataFrame(Seq(
      ("0", 1, 2, 3, 4),
      ("5", 6, 7, 8, 9)
    )).toDF("id", "a", "b", "c", "d")
  }

  test("FeatureSelector transformSchema") {
    val selector = new FeatureSelector()
      .setOutputCols(Array("id"))

    val empty = Array[StructField]()
    val emptySchema = new StructType(empty)
    evaluating {selector.transformSchema(emptySchema)} should produce [IllegalArgumentException]

    val singleField = new StructField("id", StringType)
    val singleSchema = new StructType(Array(singleField))
    val transformedSchema = selector.transformSchema(singleSchema)
    transformedSchema.fields.foreach(println)
    transformedSchema.fields should have length 1
    transformedSchema.fields(0).name should be ("id")

    val secondField = new StructField("otherName", StringType)
    val doubleSchema = new StructType(Array(singleField, secondField))
    val doubleTransformedSchema = selector.transformSchema(doubleSchema)
    doubleTransformedSchema.fields should have length 1
    doubleTransformedSchema.fields(0).name should be ("id")
  }

  test("FeatureSelector") {
    var selector = new FeatureSelector()
                   .setOutputCols(Array("id", "b", "c"))
    var rows = selector.transform(df).collect()
    rows should have length 2
    rows(0).toSeq should be (Seq("0", 2, 3))
    rows(1).toSeq should be (Seq("5", 7, 8))
  }

  after {
    sc.sparkContext.stop
    sc   = null
    conf = null
    // To avoid Akka rebinding to the same port,
    // since it doesn't unbind immediately on shutdown
    System.clearProperty("spark.master.port")
  }

}

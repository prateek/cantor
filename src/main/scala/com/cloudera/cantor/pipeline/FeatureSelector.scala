package com.cloudera.cantor.pipeline

import org.apache.spark.ml.param._
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType

// TODO: migrate
trait HasOutputCols extends Params {
  final val outputCols: StringArrayParam = new StringArrayParam(this, "outputCols", "output column names")
  final def getOuputCols: Array[String] = $(outputCols)
}

class FeatureSelector(override val uid: String)
  extends Transformer with HasOutputCols
{
  def this() = this(Identifiable.randomUID("projecter"))
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  override def transform(dataset: DataFrame): DataFrame = {
    $(outputCols) match {
      case Array(head, rest @ _*) => dataset.select(head, rest:_*)
      case _ => dataset
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    val outputColNames = $(outputCols)
    outputColNames.foreach( colName => {
      if (!schema.fieldNames.contains(colName)) {
        throw new IllegalArgumentException(s"Output column $colName does not exist.")
      }
    })
    new StructType(schema.fields.filter(field => outputColNames.contains(field.name)))
  }
}

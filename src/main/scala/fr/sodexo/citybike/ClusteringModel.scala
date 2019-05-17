package fr.sodexo.citybike

import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}

/**
  * This class represents the model.
  */
class ClusteringModel extends Serializable with Logging {

  var kMeans: KMeans = new KMeans().setK(2).setSeed(1L)
  var model: KMeansModel = null


  /**
    * Train the clustering model.
    *
    * Model is directly setup and trained on the data.
    *
    * @param data
    * @return
    */
  def fit(data: RDD[LabeledPoint]): KMeansModel = {
    model = kMeans.run(data.map(_.features))
    return model
  }


  /**
    * Predict data from input.
    * @param data
    * @return
    */
  def predict(data: RDD[Vector]): RDD[Int] = {
    model.predict(data)
  }

  /**
    * Save my model.
    * @param sc
    * @param path
    */
  def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }

  /**
    * Load my model to
    * @param path
    */
  def load(sc:SparkContext, path: String): Unit = {
    model = KMeansModel.load(sc, path)
  }

}

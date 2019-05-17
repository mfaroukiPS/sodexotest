package fr.sodexo.citybike

import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, sum}
import org.apache.spark.sql.types.IntegerType

import org.apache.spark.sql.SaveMode
import org.apache.spark._
import com.microsoft.azure.cosmosdb.spark.schema._
import com.microsoft.azure.cosmosdb.spark.CosmosDBSpark
import com.microsoft.azure.cosmosdb.spark.config.Config

import org.apache.spark.sql.functions._


import scala.util.Try

/**
  * Describes a city bike station
  * @param id
  * @param name
  * @param address
  * @param latitude
  * @param longitude
  * @param position
  */
case class CityBike(id: Int, name: String, address: String, latitude: Option[Double], longitude: Option[Double], position: String)

class CityBikeClustering(spark: SparkSession) extends Serializable with Logging {

  /**
    * Prepare the input data from raw CityBike Station data..
    * @param path
    * @return
    */
  def execute(path: String) : RDD[LabeledPoint] = {
    // Step 1 : Load raw data input :
    val df = loadData(path)
    val count = df.count() // Used for later error reporting.

    // Step 2 : Wrangle data, merge data.
    // Even if I developped name and address repair base on heuristic,
    // this is an unecessary step to perform clustering.
    // Methods for data cleaning are kept below so as their tests,
    // without being used by the actual flow.
    val preparedData = toRDD(df).map(c => (c.id, c)).reduceByKey(mergeCityBike(_,_)).cache()
    val normalizedData = filterNullCoordinates(preparedData)


    // Always interesting to check some stats on data.
    // Quick version to extract errors, would better output a structure including errors to report.
    val countedValidData = normalizedData.count()
    val discardedData = count - countedValidData
    log.info("Discarded " + discardedData + " data points missing ")
    if (discardedData > 0 ){
      preparedData.filter(x => x._2.latitude.isEmpty || x._2.longitude.isEmpty)
        .map(c => c._2.productIterator.mkString(",")).saveAsTextFile(path + "/../errors")
    }

    return normalizedData.map(c => cityBikeToLabeledPoint(c._2)).filter(_.isDefined).map(_.get)
  }


  def cityBikeToLabeledPoint(c: CityBike): Option[LabeledPoint] = {
    assert(c.id.isInstanceOf[Int])
    assert(c.latitude.isInstanceOf[Option[Double]])
    assert(c.longitude.isInstanceOf[Option[Double]])
    try {
      return Some(LabeledPoint(c.id.toDouble, Vectors.dense(c.latitude.get, c.longitude.get)))
    } catch {
      case _ => log.error("Can't build point : " + c.toString)
    }
    log.error("Can't build point : " + c.toString)
    None
  }


  /**
    * Load city bike data from file.
    *
     |-- address: string (nullable = true)
     |-- coordinates: struct (nullable = true)
     |    |-- latitude: double (nullable = true)
     |    |-- longitude: double (nullable = true)
     |-- id: integer (nullable = true)
     |-- latitude: string (nullable = true)
     |-- longitude: string (nullable = true)
     |-- name: string (nullable = true)
     |-- position: string (nullable = true)

    * @param path
    * @return
    */
  def loadData(path: String): DataFrame = {
    val df = spark.read.option("multiline", "true").json(path)
    df.withColumn("id", col("id").cast(IntegerType))
    return df
  }


  /**
    * Save data into LIBSVM file format
    * @param data
    * @param path
    */
  def saveDate(data: RDD[LabeledPoint], path: String): Unit = {
    MLUtils.saveAsLibSVMFile(data, path)
  }

  /**
    * Transform a row from data to a CityBike object.
    *
    * It takes an asumption to replace latitude by coordinates.latitude and longitude by coordinates.longitude
    * if the formers are null.
    *
    * @param r
    * @return
    */
  def rowToCityBike(r: Row): CityBike = {

    /**
      * Helper to extract to Latitude or Longitude.
      * @param r
      * @param field
      * @return
      */
    def getCoord(r: Row, field: String): Option[Double] = {
      // Find the index of the field name
      val idx: Int = r.schema.fieldIndex(field)
      val c: Option[Double] = Try(r.getAs[String](field).toDouble).toOption

      val coordinates = r.getAs[Row]("coordinates")
      if (c.isEmpty) {
        if (Option(coordinates).isDefined && !coordinates.isNullAt(coordinates.schema.fieldIndex(field))) {
          return Some(coordinates.getAs[Double](field))
        }
        return None
      }
      return c
    }

    val id: Int = r.getAs[Double]("id").toInt
    val name: String = r.getAs[String]("name")
    val address: String = r.getAs[String]("address")
    val lat: Option[Double]= getCoord(r, "latitude")
    val long: Option[Double] = getCoord(r, "longitude")
    val position: String = r.getAs[String]("position")

    return CityBike(id, name, address, lat, long, position)
  }

  /**
    * While inspecting the files, it shows that :
    * - geolocation can be expressed either as
    *   - latitude, longitude,
    *   - or an object coordinates, holding latitude, longitude.
    *
    * The goal here is to extract object coordinates correctly into the splitted lattitude, longitude
    * @param input
    * @return
    */
  def toRDD(input: DataFrame): RDD[CityBike] = {
    return input.rdd.map(rowToCityBike(_))
  }


  /**
    * At this step, we assert that both city bike stations have :
    * - Same id (by construction)
    * - Same name and address (if one is null or empty, replace by the other.)
    * - Same latitude and longitude
    *
    * @param a
    * @param b
    * @return
    */
  def mergeCityBike(a: CityBike, b: CityBike) : CityBike = {
    val lat = if (a.latitude.isEmpty) { b.latitude } else { a.latitude}
    val long = if (a.longitude.isEmpty) { b.longitude } else { a.longitude }
    val position = if (Option(a.position).isEmpty) {b.position} else {a.position}
    return CityBike(a.id, a.name, a.address, lat, long, position)
  }


  /**
    * Filter to keep only fully defined coordinates.
    * @param input
    * @return
    */
  def filterNullCoordinates(input: RDD[(Int,CityBike)]): RDD[(Int,CityBike)] = {
    return input.filter(c => c._2.latitude.isDefined && c._2.longitude.isDefined)
  }

  /**
    * If the name is missing, rebuild the nae from id and address
    *
    * Example of contract oriented function for functional programming.
    * Input contract stipulates that ID is not null.
    * Output contract defines a non null name, everything remains identical.
    *
    * The development of this method is of course overkill as such. But complex function definition
    * in particular in functional programming can take benefit of this technique, in particular while debugging.
    *
    * The asserts are transparent to performance once in production using the compiler option -Xdisable-assertions
    *
    * @param cb with id.
    * @return
    */
  def repairCityBikeName(cb: CityBike): CityBike = {
    assert(Option(cb.id).isDefined) // Input contract

    val repairedCityBike = if (Option(cb.name).isEmpty) {
      CityBike(cb.id, cb.id.toString + " - " + cb.address.toUpperCase(),
        cb.address, cb.latitude, cb.longitude, cb.position)
    } else {
      cb
    }

    assert(!repairedCityBike.name.isEmpty)
    assert(cb.id == repairedCityBike.id)

    return repairedCityBike
  }

  /**
    * Rebuild the address from the name assuming name is
    * ID - NAME
    *
    * @param cb
    * @return
    */
  def repairCityBikeAddress(cb: CityBike): CityBike = {
    if (Option(cb.address).isEmpty && Option(cb.name).isDefined) {
      val name = Try(cb.name.split("-")(1).trim).getOrElse("").toLowerCase
      val address = name.split("\\s+").map(_.capitalize).mkString(" ")
      return CityBike(cb.id, cb.name,
        address, cb.latitude, cb.longitude, cb.position)
    }
    return cb
  }




}


object CityBikeClustering extends Serializable with Logging {

  val spark: SparkSession = SparkSession
    .builder()
    .appName(getClass.getSimpleName)
    .getOrCreate()

  import spark.implicits._

  def main(args: Array[String]): Unit = {

    // Step 1 : Check parameters to launch
    if (args.length < 2) {
      log.error(
        """
          |Command : CityBikeClustering [input-files] [output-directory] [config file]
          |
          |input-files : file input path of json data.
          |output-directory : directory path.
          |
          |Path can be local or in a cloud based path (e.g. hdfs://... )
          |
        """.stripMargin)
      System.exit(-1)
    }

    val inputFile = args(0)
    val outputDir = args(1)
    // val configFile = args(2) // TODO : Optional configuration file.

    // Step 1 : Load data
    val cityBikeDataPreparation = new CityBikeClustering(spark)

    // Step 2 : Data wrangling
    val prepareData = cityBikeDataPreparation.execute(inputFile)

    // Step 3 : Record data for later use.
    // TODO : Depending on business constraints and usage, the file type has to be adapted.
    cityBikeDataPreparation.saveDate(prepareData, outputDir + "/training-data")

    // Step 4 :
    val model = new ClusteringModel()
    model.fit(prepareData)

    model.save(spark.sparkContext, outputDir + "/model")

    val predictions = model.predict(prepareData.map(_.features))
    predictions.saveAsTextFile(outputDir + "/predictions")

    // Save to cosmos
    // Configure connection to destination collection with new collection

    val endpoint = System.getProperty("endpoint")
    val masterkey = System.getProperty("masterkey")
    val secondkey = System.getProperty("secondkey")
    val writeConfig = Config(Map(
      "Endpoint" -> s"$endpoint",
      "Masterkey" -> s"$masterkey",
      "Secondkey"     -> s"$secondkey",
      "Database" -> "DB",
      "Collection" -> "predictions",
      "WritingBatchSize" -> "100"))
    predictions.toDF().write.mode(SaveMode.Overwrite).cosmosDB(writeConfig)

    spark.stop()
  }

}

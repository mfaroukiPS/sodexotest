package fr.sodexo.citybike

import com.holdenkarau.spark.testing.{DataFrameSuiteBase, SharedSparkContext, SparkSessionProvider}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.scalatest.FunSuite
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema

class CityBikeClusteringTest extends FunSuite with SharedSparkContext {

  //  |-- address: string (nullable = true)
  //  |-- coordinates: struct (nullable = true)
  //  |    |-- latitude: double (nullable = true)
  //  |    |-- longitude: double (nullable = true)
  //  |-- id: integer (nullable = true)
  //  |-- latitude: string (nullable = true)
  //  |-- longitude: string (nullable = true)
  //  |-- name: string (nullable = true)
  //  |-- position: string (nullable = true)
  val coordSchema = new StructType()
    .add("latitude", "double")
    .add("longitude", "double")

  val schema = new StructType()
    .add("id", "double")
    .add("address", "string")
    .add("name", "string")
    .add("position", "string")
    .add("latitude", "string") // need conversion to double
    .add("longitude", "string") // need conversion to double
    .add("coordinates",coordSchema)

  test("Row conversion to CityBike"){

    val executor = new CityBikeClustering(SparkSessionProvider._sparkSession)

//      "id": 109,
//      "name": "109 - MONTAGUE RD / SKINNER ST",
//      "address": "Montague Rd / Skinner St",
//      "latitude": -27.48172,
//      "longitude": 153.00436
    val row = new GenericRowWithSchema(Array(109.0, "Montague Rd / Skinner St", "109 - MONTAGUE RD / SKINNER ST", null,"-27.48172","153.00436", null), schema)
    assertResult(row.getDouble(0).toInt)(row.getAs[Int]("id"))
    val cb = executor.rowToCityBike(row)
    assertResult(109)(cb.id)
    assertResult("Montague Rd / Skinner St")(cb.address)
    assertResult("109 - MONTAGUE RD / SKINNER ST")(cb.name)
    assertResult(-27.48172)(cb.latitude.get)
    assertResult(153.00436)(cb.longitude.get)

  }

  test("Row conversion to CityBike with coordinates"){
    val executor = new CityBikeClustering(SparkSessionProvider._sparkSession)

    //      "id": 109,
    //      "name": "109 - MONTAGUE RD / SKINNER ST",
    //      "address": "Montague Rd / Skinner St",
    //      "latitude": -27.48172,
    //      "longitude": 153.00436
    val row = new GenericRowWithSchema(Array(109.0, "Montague Rd / Skinner St", "109 - MONTAGUE RD / SKINNER ST", null,null,null,
      new GenericRowWithSchema(Array(-27.48172,153.00436), coordSchema)), schema)
    assertResult(null)(row.getAs[Double]("latitude"))

    val cb = executor.rowToCityBike(row)
    assertResult(109)(cb.id)
    assertResult("Montague Rd / Skinner St")(cb.address)
    assertResult("109 - MONTAGUE RD / SKINNER ST")(cb.name)
    assertResult(-27.48172)(cb.latitude.get)
    assertResult(153.00436)(cb.longitude.get)

  }


  test("Repair city bike station name") {
    val executor = new CityBikeClustering(SparkSessionProvider._sparkSession)
    val row = new GenericRowWithSchema(Array(109.0, "Montague Rd / Skinner St", null, null,"test",null,new GenericRowWithSchema(Array(-27.48172,153.00436), coordSchema)), schema)
    val cb = executor.rowToCityBike(row)
    assertResult("109 - MONTAGUE RD / SKINNER ST")(executor.repairCityBikeName(cb).name)
  }

  test("Repair city bike station address from given name") {
    val executor = new CityBikeClustering(SparkSessionProvider._sparkSession)
    val row = new GenericRowWithSchema(Array(109.0, null, "109 - MONTAGUE RD / SKINNER ST", null,"-27.48172",null,new GenericRowWithSchema(Array(null,153.00436), coordSchema)), schema)
    val cb = executor.rowToCityBike(row)
    assertResult("Montague Rd / Skinner St")(executor.repairCityBikeAddress(cb).address)
    assertResult(-27.48172)(cb.latitude.get)
    assertResult(153.00436)(cb.longitude.get)
  }

  test("City bike station to labeled point") {
    val cb = CityBike(10, "test", "test", Some(-27.48172), Some(153.00436), "")
    val executor = new CityBikeClustering(SparkSessionProvider._sparkSession)
    val lp = executor.cityBikeToLabeledPoint(cb)
    assertResult(10.0)(lp.get.label)
    assertResult(Vectors.dense(-27.48172, 153.00436))(lp.get.features)
  }

  test("Filtering null coordinates in citybikes") {
    val rdd = sc.parallelize(Seq((10, CityBike(10, "test", "test", Some(-27.48172), Some(153.00436), "")),
      (11, CityBike(11, "test", "test", None, Some(153.00436), "")),
      (12, CityBike(12, "test", "test", Some(-27.48172), None, ""))))
    val executor = new CityBikeClustering(SparkSessionProvider._sparkSession)
    val out = executor.filterNullCoordinates(rdd)
    assertResult(1)(out.count())
  }

  test("Merge two city bikes") {
    val rdd = sc.parallelize(Seq((10, CityBike(10, "test", "test", Some(-27.48172), Some(153.00436), "")),
      (11, CityBike(11, "test", "test", None, Some(153.00436), "")),
      (11, CityBike(12, "test", "test", Some(-27.48172), None, ""))))
    val executor = new CityBikeClustering(SparkSessionProvider._sparkSession)
    val out = rdd.reduceByKey((a,b) => executor.mergeCityBike(a, b))
    assertResult(2)(executor.filterNullCoordinates(out).count())
    println(out)
  }
}

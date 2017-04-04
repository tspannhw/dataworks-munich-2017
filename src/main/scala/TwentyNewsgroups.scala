import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Created by yanboliang.
 */
object TwentyNewsgroups {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("TwentyNewsgroups").master("local[4]").getOrCreate()
    import spark.implicits._

    // In order to get faster execution times for this first example
    // we will work on a partial dataset with only 4 categories
    // out of the 20 available in the dataset
    val categories = Array("alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med")
    var training: DataFrame = spark.emptyDataFrame
    var test: DataFrame = spark.emptyDataFrame
    var isFirst: Boolean = true

    categories.foreach { category =>
      val trainingData = spark.sparkContext
        .wholeTextFiles(s"data/20news-bydate-train/$category")
        .toDF("directoryName", "content")
      val testData = spark.sparkContext
        .wholeTextFiles(s"data/20news-bydate-test/$category")
        .toDF("directoryName", "content")
      if (isFirst) {
        training = trainingData
        test = testData
        isFirst = false
      } else {
        training = training.union(trainingData)
        test = test.union(testData)
      }
    }

    val extractID = udf { directoryName: String => directoryName.split("/").last }
    val extractCategory = udf { directoryName: String => directoryName.split("/").dropRight(1).last }
    val trainingDF = training.withColumn("ID", extractID(col("directoryName")))
      .withColumn("category", extractCategory(col("directoryName")))
      .drop("directoryName")
    val testDF = test.withColumn("ID", extractID(col("directoryName")))
      .withColumn("category", extractCategory(col("directoryName")))
      .drop("directoryName")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("label")

    val tokenizer = new RegexTokenizer()
      .setInputCol("content")
      .setOutputCol("tokens")
      .setPattern("\\W")

    val swRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("words")

    val cv = new CountVectorizer()
      .setInputCol(swRemover.getOutputCol)
      .setOutputCol("wordCount")
      .setVocabSize(30000)

    val idf = new IDF()
      .setInputCol(cv.getOutputCol)
      .setOutputCol("features")

    val nb = new NaiveBayes()
      .setFeaturesCol(idf.getOutputCol)
      .setLabelCol(indexer.getOutputCol)

    val stages = Array(indexer, tokenizer, swRemover, cv, idf, nb)
    val pipeline = new Pipeline().setStages(stages)

    val paramGrid = new ParamGridBuilder()
      .addGrid(idf.minDocFreq, Array(0, 1, 2))
      .addGrid(nb.smoothing, Array(0.0, 0.5, 1.0))
      .build()

    val crossValidator = new CrossValidator()
      .setNumFolds(3)
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(new MulticlassClassificationEvaluator)

    val cvModel = crossValidator.fit(trainingDF)
    val model = cvModel.bestModel
    println(model.explainParams())

    val predictionDF = model.transform(testDF)
    val evaluator = new MulticlassClassificationEvaluator()
      .setPredictionCol(nb.getPredictionCol)
      .setLabelCol(indexer.getOutputCol)
      .setMetricName("accuracy")

    println("accuracy = " + evaluator.evaluate(predictionDF))

    spark.stop
  }
}

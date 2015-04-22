package dropout

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils._

/**
 * Testing dropout logistic regression.
 */
object example extends App {

  val sc = new SparkContext("local[4]", "dropoutTest")

  // Load and parse the data file
  val data = {
    val d = loadLibSVMFile(sc, "./src/main/resources/covtype_libsvm.csv").repartition(8).cache
    d.map(x => LabeledPoint(if(x.label == 2) 0 else x.label, x.features))
  }


  val foldData = kFold(data, 10, 1)

  var trainErr: Double = 0.0
  var trainErrDropout: Double = 0.0

  foldData.take(1).foreach{
    case (trainData, testData) =>

      // Scaler for feature scaling.
      val scaler =
        new StandardScaler(withStd = true, withMean = false).fit(trainData.map(_.features))

      // Scaler training data.
      val scaledTrainData = trainData.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features))).cache()

      // Run training algorithm to build the model
      val numIterations = 400
      val lrModel =
        LogisticRegressionWithSGD.train(trainData, numIterations, 1.0, 1.0)
      val dropOutModel =
        DropoutLogisticRegressionWithSGD.train(trainData, numIterations, 1.0, 0.5, 0.0,  1.0)


      // Scale the weights back to the same scale as the data.

      def scaleBack(model:LogisticRegressionModel) = {

        var weights = model.weights
        val numOfLinearPredictor = model.numClasses - 1
        val addIntercept = false // This is specific to this example, if you add intercept, this should be set as well.

        /**
         * Copied from [[org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm]]
         * */
        if (numOfLinearPredictor == 1) {
          weights = scaler.transform(weights)
        } else {
          /**
           * For `numOfLinearPredictor > 1`, we have to transform the weights back to the original
           * scale for each set of linear predictor. Note that the intercepts have to be explicitly
           * excluded when `addIntercept == true` since the intercepts are part of weights now.
           */
          var i = 0
          val n = weights.size / numOfLinearPredictor
          val weightsArray = weights.toArray
          while (i < numOfLinearPredictor) {
            val start = i * n
            val end = (i + 1) * n - {
              if (addIntercept) 1 else 0
            }

            val partialWeightsArray = scaler.transform(
              Vectors.dense(weightsArray.slice(start, end))).toArray

            System.arraycopy(partialWeightsArray, 0, weightsArray, start, partialWeightsArray.size)
            i += 1
          }
          weights = Vectors.dense(weightsArray)
        }

        new LogisticRegressionModel(weights, model.intercept)
      }

      val scaledLRModel = scaleBack(lrModel)
      val scaledDropoutModel = scaleBack(dropOutModel)

      // Evaluate model on training examples and compute training error
      val labelAndPreds = testData.map { point =>
        val prediction = scaledLRModel.predict(point.features)
        (point.label, prediction)
      }
      val labelAndPredsDropout = testData.map { point =>
        val prediction = scaledDropoutModel.predict(point.features)
        (point.label, prediction)
      }

      trainErr += labelAndPreds.filter(r => r._1 != r._2).count.toDouble / data.count
      trainErrDropout += labelAndPredsDropout.filter(r => r._1 != r._2).count.toDouble / data.count
  }

  println("Validation Error for LR = " + trainErr/1)
  println("Validation Error for LR with dropout = " + trainErrDropout/1)

  sc.stop()

}


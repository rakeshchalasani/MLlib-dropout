package dropout

import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression.{LabeledPoint, GeneralizedLinearAlgorithm}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD

/**
 * Train a classification model for Binary Logistic Regression
 * using Stochastic Gradient Descent with drop-out logistic regression. By default L2 regularization is used,
 * which can be changed via [[org.apache.spark.mllib.classification.LogisticRegressionWithSGD.optimizer]].
 * NOTE: Labels used in Logistic Regression should be {0, 1, ..., k - 1}
 * for k classes multi-label classification problem.
 */
class DropoutLogisticRegressionWithSGD (
    private var stepSize: Double,
    private var dropoutRate: Double,
    private var numIterations: Int,
    private var regParam: Double,
    private var miniBatchFraction: Double)
  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {

  private val gradient = new DropoutLogisticGradient(dropoutRate)
  private val updater = new SquaredL2Updater()
  override val optimizer = new DropoutGradientDescent(gradient, updater)
      .setStepSize(stepSize)
      .setDropoutRate(dropoutRate)
      .setNumIterations(numIterations)
      .setRegParam(regParam)
      .setMiniBatchFraction(miniBatchFraction)
  override protected val validators = List(DataValidators.binaryLabelValidator)


  /**
   * Construct a LogisticRegression object with default parameters: {stepSize: 1.0,
   * dropoutRate: 0.5, numIterations: 100, regParm: 0.01, miniBatchFraction: 1.0}.
   */
  def this() = this(1.0, 0.5, 100, 0.01, 1.0)

  override protected def createModel(weights: Vector, intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }
}
object DropoutLogisticRegressionWithSGD {
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double,
      dropoutRate: Double,
      miniBatchFraction: Double,
      initialWeights: Vector): LogisticRegressionModel = {
    new DropoutLogisticRegressionWithSGD(stepSize, dropoutRate, numIterations, 0.0, miniBatchFraction)
      .run(input, initialWeights)
  }

  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double,
      dropoutRate: Double,
      regParam: Double,
      miniBatchFraction: Double): LogisticRegressionModel = {
    new DropoutLogisticRegressionWithSGD(stepSize, dropoutRate, numIterations, regParam, miniBatchFraction)
      .run(input)
  }
}

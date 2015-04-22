package dropout

import org.apache.spark.Logging
import org.apache.spark.annotation.{DeveloperApi, Experimental}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.{GradientDescent, Optimizer, Updater, Gradient}
import org.apache.spark.rdd.RDD

/**
 * Class used to solve an optimization problem using Gradient Descent with dropout.
 * @param gradient Gradient function to be used.
 * @param updater Updater to be used to update weights after every iteration.
 *
 * Spark MLlib is very conservative about making things public,
 * had to re-write most of the MLlib Gradient Descent class.
 */
class DropoutGradientDescent (private var gradient: Gradient, private var updater: Updater)
  extends Optimizer with Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private var dropoutRate: Double = 0.5

  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }

  /**
   * :: Experimental ::
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  @Experimental
  def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for SGD.
   */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
   * Set the dropout rate. Default 0.5.
   */
  def setDropoutRate(p: Double): this.type = {
    this.dropoutRate = p
    this
  }

  /**
   * :: DeveloperApi ::
   * Runs gradient descent on the given training data.
   * After training, the weights are scaled with dropout probability.
   * @param data training data
   * @param initialWeights initial weights
   * @return solution vector
   */
  @DeveloperApi
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = GradientDescent.runMiniBatchSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights)
    weights.mapActive((idx:Int,value:Double) => (idx,value * dropoutRate))
  }

}

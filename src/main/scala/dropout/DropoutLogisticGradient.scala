package dropout

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.LogisticGradient
import breeze.stats.{distributions => brzDist}

/**
 * Compute gradient and loss for a multinomial logistic loss function with drop-out regularization,
 * as used in multi-class classification (it is also used in binary logistic regression).
 * Check [[LogisticGradient]] for details on multinomial logistic loss function.
 * */
@DeveloperApi
class DropoutLogisticGradient(p: Double = 0.5, numClasses: Int) extends LogisticGradient(numClasses) {

  def this(p: Double) = this(p, 2)

  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {

    // Drop-out some features randomly with probability p.
    val dropoutDist = brzDist.Bernoulli.distribution(p)
    val droppedData = data.mapActive((idx:Int, value:Double) => (idx, if(dropoutDist.draw()) value else 0.0))
    super.compute(droppedData, label, weights, cumGradient)
  }

}

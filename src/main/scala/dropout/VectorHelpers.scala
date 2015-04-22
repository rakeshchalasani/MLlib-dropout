package dropout

import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector}

import scala.collection.mutable.ArrayBuffer

/**
 * Helper functions over Spark MLlib Vectors.
 */
class VectorHelpers(v:Vector) {

  def mapActive(f: (Int, Double) => (Int, Double)):Vector = {
    if(v.isInstanceOf[DenseVector]) {
      val typedVector = v.asInstanceOf[DenseVector]
      var i = 0
      val localValuesSize = typedVector.values.size
      val localValues = typedVector.values
      val outValues = new Array[Double](localValuesSize)

      while (i < localValuesSize) {
        outValues(i) = f(i, localValues(i))._2
        i += 1
      }
      new DenseVector(outValues)
    } else {
      val typedVector = v.asInstanceOf[SparseVector]
      var i = 0
      val localValuesSize = typedVector.values.size
      val localIndices = typedVector.indices
      val localValues = typedVector.values
      var outIndices = new ArrayBuffer[Int]()
      var outValues = new ArrayBuffer[Double]()

      while (i < localValuesSize) {
        val result = f(localIndices(i), localValues(i))
        if(result._2 != 0.0) {
          outIndices += result._1
          outValues += result._2
        }
        i += 1
      }
      new SparseVector(size = typedVector.size, outIndices.toArray, outValues.toArray)
    }

  }

}


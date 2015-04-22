import org.apache.spark.mllib.linalg.Vector

package object dropout {

  implicit def vectorHelper(v:Vector):VectorHelpers = new VectorHelpers(v)
}

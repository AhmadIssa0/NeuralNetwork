
package neuralnetwork

import breeze.linalg.{Vector => _, _}
import breeze.numerics._

case class Network private(
  size: Vector[Int],
  nLayers: Int,
  weights: Vector[DenseMatrix[Double]],
  biases: Vector[DenseVector[Double]]) {


  def output(input: DenseVector[Double]): DenseVector[Double] = {
    require(input.length == size(0), "Input vector to NN of wrong size.")

    (0 until weights.length).foldLeft(input) { (a, i) =>
      sigmoid(weights(i) * a + biases(i))
    }
  }

  def backprop(input: DenseVector[Double],
    output: DenseVector[Double]
  ): (Vector[DenseMatrix[Double]], Vector[DenseVector[Double]]) = {
    require(input.length == size(0) && output.length == size(size.length - 1))

    // Computed predicted output and Z's
    val (predOutput, zs) = 
      (0 until weights.length).foldLeft((input, Vector[DenseVector[Double]]())) {
        case ((a, zs), i) =>
          val z = weights(i) * a + biases(i)
        (sigmoid(z), zs :+ z)
      }

    //println("zs: " + zs)
    def sigmoidDeriv(v: DenseVector[Double]) = sigmoid(v) :* (1. - sigmoid(v))

    // Last layer
    // cross-entropy
    val crossEntDeriv = (-(output :/ predOutput) + ((1. - output) :/ (1. - predOutput)))
    val deltaL = crossEntDeriv  :* sigmoidDeriv(zs.last)
    // MSE
    //val deltaL = (predOutput - output) :* sigmoidDeriv(zs.last)
    //println("deltaL: " + deltaL)
    //println("weights: ")
    //weights.foreach { w => println("s" + w) }
    val deltas = 
    (weights.length - 1 to 1 by -1).foldLeft(Vector[DenseVector[Double]](deltaL)) {
      (deltas, i) =>
        ((weights(i).t * deltas(0)) :* sigmoidDeriv(zs(i - 1))) +: deltas
    }

    val as = input +: zs.map(sigmoid(_))

    val gradWs = deltas.zip(as).map { case (d, a) => d * a.t }

    (gradWs, deltas)
  }

  def descend(miniBatch: Vector[(DenseVector[Double], DenseVector[Double])],
    learningRate: Double = 1.0,
    lambda: Double = 0.001 // regularization parameter
  ): Network = {

    val alpha = learningRate / miniBatch.length
    val weightDecay = (1 - learningRate * lambda / miniBatch.length)

    val decayedW = weights.map(_ :* weightDecay)
    
    val (nWeight, nBias) = 
    miniBatch.foldLeft((decayedW, biases)) {
      case ((w, b), t) =>
        val (w1, b1) = backprop(t._1, t._2)
        ((0 until w.length).toVector.map(i => w(i) - (alpha :* w1(i))),
         (0 until b.length).toVector.map(i => b(i) - (alpha :* b1(i))))
    }

    this.copy(weights = nWeight, biases = nBias)
  }

}


object Network {

  /**
    * @param size Number of nodes in each layer, starting with input layer
    *             ending with output layer.
    */
  def apply(size: Vector[Int]): Network = {
    require(size.length > 0 && size.forall(s => s > 0))

    /* Recommended in: http://stats.stackexchange.com/a/186351 */
    def randomInitialize(size_in: Int, size_out: Int): Double = {
      val r = scala.util.Random    
      val epsilon = 2*math.sqrt(6 / (size_in + size_out))
      r.nextDouble * 2 * epsilon - epsilon   
    }

    val weights = size
      .sliding(2)
      .toVector 
      .map { case Vector(n, m) =>
        DenseMatrix.tabulate(m, n)((_, _) => randomInitialize(n, m))
      }

    Network(size = size,
      nLayers = size.length,
      weights = weights,
      biases = size.tail.map(n => DenseVector.zeros[Double](n))
    )
  }
}





import breeze.linalg._

case class Network private(
  size: Vector[Int],
  nLayers: Int,
  weights: Vector[DenseMatrix[Double]],
  biases: Vector[DenseVector[Double]]) {


  def output(input: DenseVector[Double]): DenseVector[Double] = ???
}


object Network {
  def apply(size: Vector[Int]): Network = {
    require(size.forall(s => s > 0))

    Network(size = size,
      nLayers = size.length,
      weights = Vector(),
      bias = Vecotr()) 
  }

  def randomInitialize(size_in: Int, size_out: Int): double = {
   val r = scala.util.Random    
   val epsilon = 4*sqrt(6/(size_in+size_out))
   r.nextDouble*2*epsilon-epslon   
 }

}


val n = Network(Vector(3, 2, 2, 1))

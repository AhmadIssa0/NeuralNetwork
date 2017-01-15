
package neuralnetwork
import breeze.linalg.{Vector => _, _}

object Driver {

  def main(args: Array[String]): Unit = {
    var n = Network(Vector(2, 2, 1))
    println(n)
    println(n.output(DenseVector(0.,0.)))

    val training = Vector(
      (DenseVector(0.,0.), DenseVector(0.)),
      (DenseVector(0.,1.), DenseVector(1.)),
      (DenseVector(1.,0.), DenseVector(1.)),
      (DenseVector(1.,1.), DenseVector(0.))
    )

    val r = scala.util.Random
    for (i <- 0 until 500000) {
      n = n.descend(Vector(training(r.nextInt(4))),
        learningRate=0.05,
        lambda=0.002
      )
    }

    println("0, 0 -> " + n.output(DenseVector(0.,0.)))
    println("0, 1 -> " + n.output(DenseVector(0.,1.)))
    println("1, 0 -> " + n.output(DenseVector(1.,0.)))
    println("1, 1 -> " + n.output(DenseVector(1.,1.)))

    println(n)

    //println(n.backprop(DenseVector(0., 0.), DenseVector(0.)))
    //println(n)
    //println(n.output(DenseVector(Array(0., 0.))))
    //println(DenseMatrix.tabulate(3, 2)(_ + _))
    //println(10 - DenseMatrix.tabulate(3,2)(_ + _))
  }

}

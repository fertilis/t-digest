package tdigest

class Centroid(
    private var mean: Double,
    private var weight: Double
) extends Ordered[Centroid] {
  def add(sum: Double, weight: Double): Double = {
    val weight_ = this.weight
    val mean_ = this.mean
    val new_sum = sum + weight_ * mean_
    val new_weight = weight_ + weight
    this.weight = new_weight
    this.mean = new_sum / new_weight
    new_sum
  }

  override def toString(): String = {
    s"($mean, $weight)"
  }

  override def compare(that: Centroid): Int = {
    mean.compareTo(that.mean)
  }

  @inline
  def getMean: Double = {
    mean
  }

  @inline
  def getWeight: Double = {
    weight
  }

  def setWeight(value: Double): Unit = {
    weight = value
  }

  @inline
  override def clone: Centroid = {
    new Centroid(mean, weight)
  }

}

object Centroid {
  def default(): Centroid = {
    new Centroid(0.0, 1.0)
  }

  def compare(a: Centroid, b: Centroid): Int = {
    a.compare(b)
  }
}

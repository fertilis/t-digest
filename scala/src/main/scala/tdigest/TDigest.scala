package tdigest

import collection.JavaConverters._
import scala.util.control.Breaks.{break, breakable}

class TDigest(private val maxSize: Long = 100) {
  var centroids = new java.util.ArrayList[Centroid](maxSize.toInt)
  private var sum: Double = 0.0
  private var count: Double = 0.0
  private var max: Double = Double.NaN
  private var min: Double = Double.NaN

  private def updateFrom(other: TDigest) = {
    this.centroids = other.centroids
    this.sum = other.sum
    this.count = other.count
    this.max = other.max
    this.min = other.min
  }

  def mergeUnsorted(unsortedValues: Iterable[Double]): TDigest = {
    val sorted: List[Double] = unsortedValues.toSeq.sorted.toList
    mergeSorted(sorted)
  }

  def mergeSorted(sortedValues: List[Double]): TDigest = {
    if (sortedValues.isEmpty) {
      return this.clone().asInstanceOf[TDigest]
    }

    val result = new TDigest(this.maxSize);
    result.count = this.count + sortedValues.length.toDouble

    val maybeMin: Double = sortedValues.head
    val maybeMax: Double = sortedValues.last

    if (this.count > 0.0) {
      result.min = this.min.min(maybeMin)
      result.max = this.max.max(maybeMax)
    } else {
      result.min = maybeMin
      result.max = maybeMax
    }

    val compressed = new java.util.ArrayList[Centroid](this.maxSize.toInt)

    var kLimit: Double = 1.0
    var qlimitTimesCount: Double = TDigest.kToQ(kLimit, this.maxSize.toDouble) * result.count
    kLimit += 1.0

    val centroidsIterator: scala.collection.BufferedIterator[Centroid] = this.centroids.iterator().asScala.buffered
    val sortedValuesIterator: scala.collection.BufferedIterator[Double] = sortedValues.iterator.buffered

    var curr: Centroid = if (centroidsIterator.hasNext) {
      val c: Centroid = centroidsIterator.head
      val curr2: Double = sortedValuesIterator.head
      if (c.getMean < curr2) {
        centroidsIterator.next().clone()
      } else {
        new Centroid(sortedValuesIterator.next(), 1.0)
      }
    } else {
      new Centroid(sortedValuesIterator.next(), 1.0)
    }

    var weightSoFar: Double = curr.getWeight

    var sumsToMerge: Double = 0.0
    var weightsToMerge: Double = 0.0

    while (centroidsIterator.hasNext || sortedValuesIterator.hasNext) {
      val next: Centroid = if (centroidsIterator.hasNext) {
        val c: Centroid = centroidsIterator.head
        if (!sortedValuesIterator.hasNext || c.getMean < sortedValuesIterator.head) {
          centroidsIterator.next().clone()
        } else {
          new Centroid(sortedValuesIterator.next(), 1.0)
        }
      } else {
        new Centroid(sortedValuesIterator.next(), 1.0)
      }

      val next_sum: Double = next.getMean * next.getWeight
      weightSoFar += next.getWeight

      if (weightSoFar <= qlimitTimesCount) {
        sumsToMerge += next_sum
        weightsToMerge += next.getWeight
      } else {
        result.sum = result.sum + curr.add(sumsToMerge, weightsToMerge)
        sumsToMerge = 0.0
        weightsToMerge = 0.0

        compressed.add(curr.clone());
        qlimitTimesCount = TDigest.kToQ(kLimit, this.maxSize.toDouble) * result.count
        kLimit += 1.0
        curr = next
      }
    }

    result.sum = result.sum + curr.add(sumsToMerge, weightsToMerge)
    compressed.add(curr);
    compressed.sort(Centroid.compare)

    result.centroids = compressed
    updateFrom(result)
    result
  }

  def mean(): Double = {
    if (count > 0.0) {
      sum / count
    } else {
      0.0
    }
  }

  /**
   * To estimate the value located at `q` quantile
   */
  def estimateQuantile(q: Double): Double = {
    if (centroids.isEmpty) {
      return 0.0
    }
    val rank: Double = q * count

    var pos: Int = 0
    var t: Double = 0d

    if (q > 0.5) {
      if (q >= 1.0) {
        return this.max
      }

      pos = 0
      t = count

      breakable {
        for ((centroid, k) <- centroids.asScala.zipWithIndex.reverse) {
          t -= centroid.getWeight
          if (rank >= t) {
            pos = k
            break
          }
        }
      }
    } else {
      if (q <= 0.0) {
        return this.min
      }

      pos = centroids.size() - 1
      t = 0.0

      breakable {
        for ((centroid, k) <- centroids.asScala.zipWithIndex) {
          if (rank < t + centroid.getWeight) {
            pos = k
            break
          }

          t += centroid.getWeight
        }
      }
    }

    var delta: Double = 0.0
    var minVal: Double = this.min
    var maxVal: Double = this.max

    if (centroids.size() > 1) {
      if (pos == 0) {
        delta = centroids.get(pos + 1).getMean - centroids.get(pos).getMean
        maxVal = centroids.get(pos + 1).getMean
      } else if (pos == (centroids.size() - 1)) {
        delta = centroids.get(pos).getMean - centroids.get(pos - 1).getMean
        minVal = centroids.get(pos - 1).getMean
      } else {
        delta = (centroids.get(pos + 1).getMean - centroids.get(pos - 1).getMean) / 2.0
        minVal = centroids.get(pos - 1).getMean
        maxVal = centroids.get(pos + 1).getMean
      }
    }

    val value = centroids.get(pos).getMean + ((rank - t) / centroids.get(pos).getWeight - 0.5) * delta
    TDigest.clamp(value, minVal, maxVal)
  }

}

object TDigest {
  private def kToQ(k: Double, d: Double): Double = {
    val k_div_d = k / d;
    if (k_div_d >= 0.5) {
      val base = 1.0 - k_div_d;
      1.0 - 2.0 * base * base
    } else {
      2.0 * k_div_d * k_div_d
    }
  }

  private def clamp(v: Double, lo: Double, hi: Double): Double = {
    if (v > hi) {
      hi
    } else if (v < lo) {
      lo
    } else {
      v
    }
  }

}

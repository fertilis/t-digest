# Incremental Calculation of Quantile

Purpose: to have identical calculation in python and scala.

## Reference implementations: 

+ C++ <https://github.com/facebook/folly/blob/main/folly/stats/TDigest.cpp>

+ Rust: <https://github.com/MnO2/t-digest>


## Notes

Rust project was taken as a base implementation. It is itself a translation of C++ implementation.

I have abridged rust code a bit.

Python and scala implementations are my translations from rust code.

In python, number of centroids is hardcoded to 100.

To imitate quantile calculation in a rolling window use `trim_weights()`.


## Performance

Tested on uniform distributions: 1 million updates with a single value

+ Rust: 480 ns 

+ Scala: 1.4 us

+ Python (numba): 28 us


## Examples

```rust
use tdigest::TDigest;

fn main() {
    let mut t = TDigest::new_with_size(100);
    for i in 0..1000 {
        t = t.merge_sorted(vec![*i as f64]);
    }
    for q in 0..10 {
        let q: f64 = q as f64 / 10.0;
        println!("{}, {:?}", q, t.estimate_quantile(q));
    }
}
```

```scala
import tdigest.TDigest

object Main extends App {
  var d = new TDigest(100)
  (0 until 1000).foreach(x => d.mergeSorted(List(x.toDouble)))
  (0 until 10).foreach(q => {
    val q2 = q.toDouble / 10.0
    val v = d.estimateQuantile(q2)
    println(s"$q2, $v")
  })
}
```


```python
import tdigest

def main():
    d = tdigest.new()
    for i in range(1000):
        tdigest.add_value(d, i)
    for q in range(10):
        print(q / 10, tdigest.quantile(d, q / 10))

if __name__ == "__main__":
    main()
```


## Other Links:

+ <https://dataorigami.net/2015/03/19/Percentile-and-Quantile-Estimation-of-Big-Data-The-t-Digest.html>

+ <https://www.researchgate.net/profile/Frank-Klawonn/publication/225662182_Incremental_quantile_estimation/links/0fcfd5081866a99968000000/Incremental-quantile-estimation.pdf>

+ <https://github.com/tdunning/t-digest/tree/main>

+ <https://github.com/CamDavidsonPilon/tdigest/tree/master>

| API | Operator | Description |
| --- | --- | --- |
| `TrecRun(results)` | | Create a `TrecRun` object from a dictionary of results or a path to a run file in TREC format. |
| `add(self, other)`, `subtract`, `multiply`, `divide` | `+`, `-`, `*`, `/` | Perform the given operation between self's document scores and `other`, which can be a `TrecRun` or a scalar. |
| `topk(self, k)` | `%` | Retain only the top-k documents for each qid after sorting by score. |
| `intersect(self, other)` | `&` | Retain only the queries and documents that appear in both `self` and `other`. |
| `concat(self, other)` | | Concat the documents in `other` and `self`, with those in `other` appearing at the end. Their scores will be modified to accomplish this.  |
| `normalize(self, method='rr')` | | Normalize scores in self using RRF (`rr`), sklearn's min-max scaling (`minmax`), or sklearn's scaling (`standard`).|
| `write_trec_run(self, outf)` | | Write `self` to `outfn` in TREC format.|
| `evaluate(self, qrels, metrics, return_average=True)` | | Compute `metrics` for `self` using `qrels` and return either the average metric or a dict mapping metric names to their values for each QID. |


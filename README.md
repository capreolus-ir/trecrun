[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Worfklow](https://github.com/capreolus-ir/trecrun/workflows/pytest/badge.svg)](https://github.com/capreolus-ir/trecrun/actions)
[![PyPI version fury.io](https://badge.fury.io/py/trecrun.svg)](https://pypi.python.org/pypi/trecrun/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

# TRECRun

`TRECRun` is a library for working with TREC run files, with an API heavily inspired by PyTerrier's pipeline operators.

| API | Operator | Description |
| --- | --- | --- |
| `TRECRun(results)` | | Create a `TRECRun` object from a dictionary of results or a path to a run file in TREC format. |
| `add(self, other)`, `subtract`, `multiply`, `divide` | `+`, `-`, `*`, `/` | Perform the given operation between self's document scores and `other`, which can be a `TRECRun` or a scalar. |
| `topk(self, k)` | `%` | Retain only the top-k documents for each qid after sorting by score. |
| `intersect(self, other)` | `&` | Retain only the queries and documents that appear in both `self` and `other`. |
| `concat(self, other)` | | Concat the documents in `other` and `self`, with those in `other` appearing at the end. Their scores will be modified to accomplish this.  |
| `normalize(self, method='rr')` | | Normalize scores in self using RRF (`rr`), sklearn's min-max scaling (`minmax`), or sklearn's scaling (`standard`).|
| `write_trec_run(self, outf)` | | Write `self` to `outfn` in TREC format.|
| `evaluate(self, qrels, metrics, return_average=True)` | | Compute `metrics` for `self` using `qrels` and return either the average metric or a dict mapping metric names to their values for each QID. |


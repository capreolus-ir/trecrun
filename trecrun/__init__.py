import operator
from copy import deepcopy

import ir_measures
import sklearn.preprocessing
import smart_open

__version__ = "0.1.0"

DEFAULT_METRICS = [
    "P@1",
    "P@5",
    "P@10",
    "P@20",
    "Judged@10",
    "Judged@20",
    "Judged@50",
    "AP@100",
    "AP@1000",
    "nDCG@5",
    "nDCG@10",
    "nDCG@20",
    "Recall@100",
    "Recall@1000",
    "RR",
]


class TrecRun:
    # hashlib.md5(json.dumps(mrl.results, sort_keys=True).encode()).hexdigest()

    def __init__(self, results):
        if isinstance(results, dict):
            # use comprehension to ensure copy
            self.results = {str(qid): {docid: score for docid, score in results[qid].items()} for qid in results}
        else:
            self.results = {}
            with smart_open.open(results) as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) > 0:
                        qid, _, docid, rank, score = fields[:5]
                        score = float(score)
                        self.results.setdefault(qid, {})

                        if docid in self.results[qid]:
                            score = max(score, self.results[qid][docid])
                        self.results[qid][docid] = score

            if not self.results:
                raise IOError("provided path contained no results: %s" % results)

    def _arithmetic_op(self, other, op):
        if isinstance(other, TrecRun):
            try:
                results = {
                    qid: {docid: op(score, other.results[qid][docid]) for docid, score in self.results[qid].items()}
                    for qid in self.results
                }
            except KeyError:
                raise ValueError(
                    "both TrecRuns must contain the same qids and docids; perhaps you should intersect or concat first?"
                )
        else:
            scalar = other
            results = {qid: {docid: op(score, scalar) for docid, score in self.results[qid].items()} for qid in self.results}

        return TrecRun(results)

    def add(self, other):
        return self._arithmetic_op(other, operator.add)

    def subtract(self, other):
        return self._arithmetic_op(other, operator.sub)

    def multiply(self, other):
        return self._arithmetic_op(other, operator.mul)

    def divide(self, other):
        return self._arithmetic_op(other, operator.truediv)

    def topk(self, k):
        results = {}
        for qid, docscores in self.results.items():
            if len(docscores) > k:
                results[qid] = dict(sorted(docscores.items(), key=lambda x: x[1], reverse=True)[:k])
            else:
                results[qid] = docscores.copy()

        return TrecRun(results)

    def intersect(self, other):
        if not isinstance(other, TrecRun):
            raise NotImplementedError()

        shared_results = {
            qid: {docid: score for docid, score in self.results[qid].items() if docid in other.results[qid]}
            for qid in self.results.keys() & other.results.keys()
        }
        return TrecRun(shared_results)

    def qids(self):
        return set(self.results.keys())

    def union_qids(self, other, shared_qids="disallow"):
        if not isinstance(other, TrecRun):
            raise NotImplementedError()

        if shared_qids == "disallow":
            if self.qids().intersection(other.qids()):
                raise ValueError("inputs share some qids but shared_qids='disallow'")

            new_results = deepcopy(self.results)
            new_results.update(deepcopy(other.results))
        else:
            raise NotImplementedError("only disallow is implemented")

        return TrecRun(new_results)

    def concat(self, other):
        results = {qid: {docid: score for docid, score in self.results[qid].items()} for qid in self.results}
        new_results = {
            qid: {docid: score for docid, score in other.results[qid].items() if docid not in self.results[qid]}
            for qid in other.results
            if qid in self.results
        }

        for qid in new_results:
            if len(new_results[qid]) == 0:
                continue

            mn, mx = min(other[qid]), max(other[qid])
            newmx = min(results[qid]) - 1e-3
            newmn = newmx - (mx - mn)
            a = (newmx - newmn) / (mx - mn)
            b = mx - a * mx

            for docid, score in new_results[qid].items():
                results[qid][docid] = a * score + b

        return TrecRun(results)

    def difference(self, other):
        results = {
            qid: {docid: score for docid, score in self.results[qid].items() if docid not in other.results.get(qid, {})}
            for qid in self.results
        }
        return TrecRun(results)

    def normalize(self, method="rr"):
        normalization_funcs = {
            "minmax": sklearn.preprocessing.minmax_scale,
            "standard": sklearn.preprocessing.scale,
        }

        if method == "rr":
            sorted_results = {
                qid: sorted(
                    ((docid, score) for docid, score in self.results[qid].items()),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for qid in self.results
            }
            results = {
                qid: {docid: 1 / (60 + idx + 1) for idx, (docid, old_score) in enumerate(sorted_results[qid])}
                for qid in sorted_results
            }
        elif method in normalization_funcs:
            results = {qid: {} for qid in self.results}
            for qid in self.results:
                docids, scores = zip(*self.results[qid].items())
                results[qid] = dict(zip(docids, normalization_funcs[method](scores)))
        else:
            raise ValueError(f"unknown method: {method}")

        return TrecRun(results)

    def __getitem__(self, k):
        # TODO is it ok to NOT return a copy here?
        return self.results[k]

    def __and__(self, other):
        return self.intersect(other)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        return (-self).add(other)

    def __truediv__(self, other):
        return self.divide(other)

    def __neg__(self):
        return self.multiply(-1)

    def __len__(self):
        return sum(len(x) for x in self.results.values())

    def __eq__(self, other):
        if isinstance(other, TrecRun):
            return self.results == other.results
        return NotImplemented

    def write_trec_run(self, outfn, tag="trecrun"):
        preds = self.results
        count = 0
        with open(outfn, "wt") as outf:
            qids = sorted(preds.keys())
            for qid in qids:
                rank = 1
                for docid, score in sorted(preds[qid].items(), key=lambda x: x[1], reverse=True):
                    print(f"{qid} Q0 {docid} {rank} {score} {tag}", file=outf)
                    rank += 1
                    count += 1

    def remove_unjudged_documents(self, qrels):
        results = {
            qid: {docid: score for docid, score in self.results[qid].items() if docid in qrels[qid]} for qid in self.results
        }
        return TrecRun(results)

    def evaluate(
        self,
        qrels,
        metrics=DEFAULT_METRICS,
        return_average=True,
    ):
        metrics = [ir_measures.parse_measure(metric) if isinstance(metric, str) else metric for metric in metrics]
        if return_average:
            d = ir_measures.calc_aggregate(metrics, qrels, self.results)
            # convert from objects to metric names
            d = {str(metric): v for metric, v in d.items()}
        else:
            d = {}
            for val in ir_measures.iter_calc(metrics, qrels, self.results):
                d.setdefault(val.query_id, {})[str(val.measure)] = val.value

        return d

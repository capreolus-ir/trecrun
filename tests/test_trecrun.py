import pytest

from trecrun import TRECRun


@pytest.fixture
def simple_run(tmp_path):
    run = """1 Q0 123 1 10
             1 Q0 124 2 9
             2 Q0 125 1 9"""
    rundict = {"1": {"123": 10, "124": 9}, "2": {"125": 9}}

    runfn = tmp_path / "simple_run"
    with open(runfn, "wt") as outf:
        outf.write(run)

    return runfn, rundict


def test_basic(simple_run):
    runfn, rundict = simple_run

    run = TRECRun(runfn)
    assert run["1"] == rundict["1"]
    assert run["2"] == rundict["2"]

    run2 = TRECRun(rundict)
    assert run2.results == rundict

    newrundict = {"1": {"123": 10 * 3 / 4 + 5 - 6, "124": 9 * 3 / 4 + 5 - 6}, "2": {"125": 9 * 3 / 4 + 5 - 6}}
    assert (run2 * 3 / 4 + 5 - 6).results == newrundict

    assert (run2 * 2).results == (run2 + run2).results

    shortrun2 = run2.topk(2)
    assert len(shortrun2["1"]) == 2
    assert len(shortrun2["2"]) == 1

    shortrun1 = run2.topk(1)
    assert len(shortrun1["1"]) == 1
    assert len(shortrun1["2"]) == 1

    assert shortrun1.intersect(TRECRun({"1": shortrun2["1"]})).results == {"1": {"123": 10}}


def test_evaluate(simple_run):
    runfn, rundict = simple_run
    qrels = {"1": {"123": 1, "124": 0}}
    run = TRECRun(rundict)

    metrics = run.evaluate(qrels)
    assert metrics["P@1"] == 1.0
    assert metrics["P@5"] == 0.2
    assert metrics["Judged@10"] == 0.2
    assert metrics["RR"] == 1.0

    qid_metrics = run.evaluate(qrels, return_average=False)
    assert qid_metrics["1"] == metrics

    metrics2 = run.evaluate({"1": {"123": 0, "124": 1}})
    assert metrics2["P@1"] == 0.0
    assert metrics2["P@5"] == 0.2
    assert metrics2["Judged@10"] == 0.2
    assert metrics2["RR"] == 0.5


def test_cache_hash(simple_run):
    runfn, rundict = simple_run
    copy1 = TRECRun(runfn)
    copy2 = TRECRun(rundict)
    assert copy1.cache_hash() == copy2.cache_hash()
    # test that query and doc IDs are normalized to strings
    assert TRECRun({1: {123: 10, 124: 9}, 2: {125: 9}}).cache_hash() == copy1.cache_hash()

    assert TRECRun({"1": {"123": 9, "124": 10}, "2": {"125": 9}}).cache_hash() != copy1.cache_hash()
    assert TRECRun({"1": {"123": 10, "124": 9}, "2": {"125": 0.9}}).cache_hash() != copy1.cache_hash()
    assert TRECRun({"1": {"123": 10, "124": 9}, "2": {"125": 9}, "2": {"0": 0}}) != copy1.cache_hash()

    assert (copy1 + 1).cache_hash() != copy2.cache_hash()
    assert copy1.topk(1).cache_hash() != copy2.cache_hash()
    assert copy1.topk(2).cache_hash() == copy2.cache_hash()
    assert copy1.normalize().cache_hash() != copy2.cache_hash()

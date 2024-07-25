import pytest

from trecrun import TRECRun


@pytest.fixture
def runfn(tmp_path):
    run = """1 Q0 123 1 10
             1 Q0 124 2 9
             2 Q0 125 1 9"""

    runfn = tmp_path / "simple_run"
    with open(runfn, "wt") as outf:
        outf.write(run)

    return runfn


@pytest.fixture
def rundict():
    return {"1": {"123": 10, "124": 9}, "2": {"125": 9}}


@pytest.fixture
def run(rundict):
    return TRECRun(rundict)


@pytest.fixture
def runmid():
    return TRECRun({"1": {"123": 10, "124": 9, "mid": 9.5}, "2": {"125": 9}})


def test_load(runfn, rundict):
    run = TRECRun(runfn)
    assert run["1"] == rundict["1"]
    assert run["2"] == rundict["2"]

    run2 = TRECRun(rundict)
    assert run2.results == rundict
    assert run == run2
    assert run.cache_hash() == run2.cache_hash()


def test_arithmetic(run):
    newrundict = {"1": {"123": 10 * 3 / 4 + 5 - 6, "124": 9 * 3 / 4 + 5 - 6}, "2": {"125": 9 * 3 / 4 + 5 - 6}}
    assert (run * 3 / 4 + 5 - 6).results == newrundict

    assert (run * 2).results == (run + run).results
    assert (run * 2).cache_hash() == (run + run).cache_hash()


def test_topk(run):
    shortrun = run.topk(2)
    assert len(shortrun["1"]) == 2
    assert len(shortrun["2"]) == 1

    shortrun1 = run.topk(1)
    assert len(shortrun1["1"]) == 1
    assert len(shortrun1["2"]) == 1

    assert shortrun.cache_hash() != shortrun1.cache_hash()


def test_intersect(run):
    shortrun = run.topk(2)
    shortrun1 = run.topk(1)
    assert shortrun1.intersect(TRECRun({"1": shortrun["1"]})).results == {"1": {"123": 10}}


def test_get_qids(run):
    assert run.qids() == {"1", "2"}


def test_concat(run, runmid):
    both = run.concat(runmid)
    assert len(both["1"]) == 3
    assert both["1"]["123"] > both["1"]["mid"]
    assert both["1"]["124"] > both["1"]["mid"]

    bigboth = run.concat(TRECRun({"1": {"123": 20, "124": 900}, "2": {"125": 100}}))
    assert bigboth["1"]["123"] > bigboth["1"]["124"]
    assert bigboth["2"]["125"] == 9


def test_difference(run, runmid):
    diff0 = run.difference(runmid)
    assert len(diff0.results) == 0

    diff1 = runmid.difference(run)
    assert len(diff1.results) == 1
    assert diff1["1"]["mid"] == 9.5


def test_normalize(runmid):
    rrf = runmid.normalize()
    assert rrf["1"]["123"] == 1 / 1
    assert rrf["1"]["mid"] == 1 / 2
    assert rrf["1"]["124"] == 1 / 3
    assert rrf["2"]

    minmax = runmid.normalize("minmax")
    assert minmax["1"]["123"] == 1
    assert minmax["1"]["124"] == 0
    assert minmax["1"]["mid"] == 0.5
    assert minmax["2"]

    standard = runmid.normalize("standard")
    assert standard["1"]["123"] > 1.22
    assert standard["1"]["124"] < -1.22
    assert standard["1"]["mid"] == 0
    assert standard["2"]


def test_remove_unjudged_documents(run, rundict):
    # return {"1": {"123": 10, "124": 9}, "2": {"125": 9}}
    qrels = {"1": {"123": 1, "124": 0}}
    new = run.remove_unjudged_documents(qrels)
    new.qids() == {"1"}

    for docid, score in new["1"].items():
        assert score == rundict["1"][docid]


def test_write_trec_run(runfn):
    run = TRECRun(runfn)
    newrunfn = str(runfn) + ".newrun"
    run.write_trec_run(newrunfn)

    newrun = TRECRun(newrunfn)
    assert run.results == newrun.results
    assert run == newrun


def test_evaluate(run):
    qrels = {"1": {"123": 1, "124": 0}}

    metrics = run.evaluate(qrels)
    assert metrics["P@1"]["1"] == 1.0
    assert metrics["P@5"]["1"] == 0.2
    assert metrics["Judged@10"]["1"] == 1.0
    assert metrics["RR"]["1"] == 1.0

    assert metrics["P@1"]["1"] == metrics["P@1"]["mean"]

    metrics2 = run.evaluate({"1": {"123": 0, "124": 1}, "2": {"125": 1}})
    assert metrics2["P@1"]["1"] == 0.0
    assert metrics2["P@5"]["1"] == 0.2
    assert metrics2["Judged@10"]["1"] == 1.0
    assert metrics2["RR"]["1"] == 0.5

    assert metrics2["P@1"]["mean"] == 0.5
    assert metrics2["P@5"]["mean"] == 0.2
    assert metrics2["Judged@10"]["mean"] == 1.0
    assert metrics2["RR"]["mean"] == 0.75


def test_cache_hash(runfn, rundict):
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


def test_json(tmp_path):
    run = TRECRun({"q1": {"d1": 1, "d2": 2}})

    outfn = tmp_path / "out.json"
    run.write_json_run(outfn)

    r2 = TRECRun(outfn)
    assert run.results == r2.results

import pytest

from trecrun import TrecRun


def test_load(tmp_path):
    run = """1 Q0 123 1 10
             1 Q0 124 2 9
             2 Q0 125 1 9"""
    rundict = {"1": {"123": 10, "124": 9}, "2": {"125": 9}}

    with open(tmp_path / "run", "wt") as outf:
        outf.write(run)

    run = TrecRun(tmp_path / "run")
    assert run["1"] == rundict["1"]
    assert run["2"] == rundict["2"]

    run2 = TrecRun(rundict)
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

    assert shortrun1.intersect(TrecRun({"1": shortrun2["1"]})).results == {"1": {"123": 10}}

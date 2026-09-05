import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "compare_embeddings", Path(__file__).parent.parent / "scripts/compare_embeddings.py"
)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def test_identical_vectors_need_no_reindex():
    label, advice = compare.verdict([1.0, 0.9999, 1.0], overlap=1.0)
    assert label == "identical"
    assert "without re-indexing" in advice


def test_small_numeric_drift_is_flagged_but_not_fatal():
    # What quantisation alone looks like: Q8_0 GGUF against fp16 safetensors.
    label, advice = compare.verdict([0.997, 0.998, 0.996], overlap=1.0)
    assert label == "close"
    assert "quantisation" in advice


def test_a_pooling_or_prompt_mismatch_reads_as_different():
    label, advice = compare.verdict([0.62, 0.71, 0.55], overlap=0.3)
    assert label == "different"
    assert "re-index" in advice


def test_one_bad_input_is_enough_to_stop_a_clean_verdict():
    # The minimum drives it: an average hides the case that breaks.
    assert compare.verdict([1.0, 1.0, 1.0, 0.40], overlap=0.9)[0] == "different"


def test_neighbour_overlap_sees_ordering_that_cosine_alone_misses():
    a = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    ranked = compare.neighbour_overlap(a, k=1)
    assert ranked[0] == [1] and ranked[1] == [0]


def test_cosine_ignores_magnitude():
    assert compare.cosine([1.0, 0.0], [4.0, 0.0]) == 1.0
    assert compare.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

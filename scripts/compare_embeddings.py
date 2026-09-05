"""Compare two embedding endpoints vector-for-vector.

The question this answers is not "is the new one faster" but "are these the same
embeddings" — because if they are not, every vector already in the index is
invalid and the migration includes a re-index.

Two endpoints serving the same weights can still disagree: pooling strategy,
instruction prefix, quantisation, dimension truncation and padding all sit
outside the forward pass, and each runtime has its own defaults for them. None
of that raises an error. You get plausible unit vectors that retrieve worse.

    uv run python scripts/compare_embeddings.py \\
      --baseline https://text-embed.model.aeriumhq.com --baseline-key $KEY_A \\
      --candidate https://text-embed-vllm.model.aeriumhq.com --candidate-key $KEY_B \\
      --corpus queries.txt
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import httpx

# Corresponding vectors above this are the same embedding to within numerics.
IDENTICAL = 0.999
# Below this, something structural differs — pooling, prompt, or dimensions.
STRUCTURAL = 0.95

DEFAULT_CORPUS = [
    "wireless noise cancelling headphones",
    "3 phase transformer 500 kva price",
    "how do I reset my password",
    "lightweight running shoes for wide feet",
    "annual report 2025 revenue breakdown",
    "replacement filter cartridge",
    "same day delivery within city limits",
    "does this come with a warranty",
]


def embed(client: httpx.Client, base: str, key: str, model: str, texts: list[str]) -> list[list[float]]:
    response = client.post(
        f"{base.rstrip('/')}/v1/embeddings",
        headers={"Authorization": f"Bearer {key}"} if key else {},
        json={"model": model, "input": texts},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return [row["embedding"] for row in sorted(data, key=lambda r: r.get("index", 0))]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def neighbour_overlap(vectors: list[list[float]], k: int) -> list[list[int]]:
    """Which items each item considers closest. Retrieval only cares about this
    ordering, so two encoders can differ numerically and still rank the same."""
    ranked = []
    for i, v in enumerate(vectors):
        scored = [(cosine(v, w), j) for j, w in enumerate(vectors) if j != i]
        scored.sort(reverse=True)
        ranked.append([j for _, j in scored[:k]])
    return ranked


def verdict(pairwise: list[float], overlap: float) -> tuple[str, str]:
    worst = min(pairwise)
    if worst >= IDENTICAL:
        return "identical", "same contract; migrate without re-indexing"
    if worst >= STRUCTURAL:
        return (
            "close",
            "numerics only (quantisation or kernels). Check the neighbour overlap "
            "and a real retrieval metric before trusting the existing index",
        )
    return (
        "different",
        "structural mismatch — pooling, instruction prefix or dimensions. "
        "Align those, or plan a re-index",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="endpoint whose vectors are already indexed")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-key", default="")
    parser.add_argument("--candidate-key", default="")
    parser.add_argument("--baseline-model", default="Qwen3-Embedding-0.6B")
    parser.add_argument("--candidate-model", default="Qwen3-Embedding-0.6B")
    parser.add_argument("--corpus", type=Path, help="one text per line; uses a small default if unset")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    texts = (
        [line.strip() for line in args.corpus.read_text().splitlines() if line.strip()]
        if args.corpus
        else DEFAULT_CORPUS
    )
    if len(texts) < 2:
        print("need at least two texts to compare", file=sys.stderr)
        return 2

    with httpx.Client() as client:
        left = embed(client, args.baseline, args.baseline_key, args.baseline_model, texts)
        right = embed(client, args.candidate, args.candidate_key, args.candidate_model, texts)

    if len(left) != len(right):
        print(f"count mismatch: {len(left)} vs {len(right)}", file=sys.stderr)
        return 1
    dims = (len(left[0]), len(right[0]))
    if dims[0] != dims[1]:
        print(
            f"dimension mismatch: {dims[0]} vs {dims[1]} — one side is truncating "
            "(Matryoshka) or serving a different model. Not comparable.",
            file=sys.stderr,
        )
        return 1

    pairwise = [cosine(a, b) for a, b in zip(left, right)]
    ranked_left = neighbour_overlap(left, args.top_k)
    ranked_right = neighbour_overlap(right, args.top_k)
    overlaps = [
        len(set(a) & set(b)) / len(a) for a, b in zip(ranked_left, ranked_right) if a
    ]
    overlap = statistics.mean(overlaps) if overlaps else 0.0
    label, advice = verdict(pairwise, overlap)

    if args.json:
        print(json.dumps({
            "texts": len(texts), "dimensions": dims[0], "verdict": label, "advice": advice,
            "cosine": {"min": min(pairwise), "mean": statistics.mean(pairwise), "max": max(pairwise)},
            "top_k": args.top_k, "neighbour_overlap": overlap,
        }, indent=2))
        return 0 if label != "different" else 1

    print(f"{len(texts)} texts, {dims[0]} dimensions\n")
    print(f"  cosine(baseline[i], candidate[i])")
    print(f"    min   {min(pairwise):.6f}")
    print(f"    mean  {statistics.mean(pairwise):.6f}")
    print(f"    max   {max(pairwise):.6f}\n")
    print(f"  top-{args.top_k} neighbour overlap: {overlap:.1%}")
    print("    how often both encoders agree on what is nearest — this is what")
    print("    retrieval actually depends on\n")
    print(f"  verdict: {label} — {advice}")

    worst = sorted(range(len(pairwise)), key=lambda i: pairwise[i])[:3]
    if pairwise[worst[0]] < IDENTICAL:
        print("\n  least similar inputs:")
        for i in worst:
            print(f"    {pairwise[i]:.6f}  {texts[i][:60]}")
    return 0 if label != "different" else 1


if __name__ == "__main__":
    raise SystemExit(main())

# Fine-tune example

Fine-tunes the local LLM (Qwen-2.5-3B) on the SDR agent's style using **LoRA**,
then serves the result through the same simple-local config. Shows the whole
loop end to end:

```
seeds + sdr transcripts  ->  MLX LoRA  ->  fuse  ->  GGUF  ->  make serve
```

Training is a **separate offline step** — llama.cpp only does inference. The
output is just a new GGUF that the existing server loads.

## Requirements

- **Apple Silicon** — MLX trains on the Metal GPU. (On other hardware, fine-tune
  with PEFT/Unsloth on a CUDA GPU instead; the convert-to-GGUF step is the same.)
- `uv` pulls `mlx-lm` on demand. The convert step clones llama.cpp for its
  `convert_hf_to_gguf.py` and pulls torch/transformers — it's heavy.
- Disk/time: the base model is ~6 GB; a short LoRA run is minutes on an M-series.

## Run

```bash
make run EXAMPLE=finetune                 # dataset -> LoRA -> fuse -> GGUF
make serve CONFIG=examples/finetune/config.yml
make run EXAMPLE=chat                     # talk to the fine-tuned model
```

Like the other examples, `make run EXAMPLE=finetune` just runs this folder's
`run.py` — here it orchestrates a build pipeline rather than calling a server, so
ignore the "LLM server not reachable" note; no server is needed for this step.
`run.py` is four steps (see the file):

1. `build_dataset.py` — turns the seed objection/response pairs **plus your
   `examples/sdr/transcripts/*.json`** into `data/{train,valid}.jsonl` (MLX chat
   format). More real calls → better data.
2. `mlx_lm.lora --train` — LoRA fine-tune on the Metal GPU → `adapters/`.
3. `mlx_lm.fuse` — merge the adapter into the base weights → `merged/`.
4. `convert_hf_to_gguf.py` — → `qwen-sdr-lora.gguf`, which `config.yml` serves.

## Caveats (it's a worked example, not magic)

- **Data quantity matters.** The ~16 seed pairs get a run going, but meaningful
  gains need dozens–hundreds of good examples. Your transcripts are the real
  fuel — run more calls first.
- **Prompt/RAG first.** For script-following, a good system prompt often beats
  fine-tuning. Reach for this when you want consistent *style/format* or to shrink
  a large prompt.
- **Version drift.** `mlx-lm` and llama.cpp flags change; if a step errors, check
  `mlx_lm.lora --help` and the current `convert_hf_to_gguf.py` usage.
- **LoRA adapters can be served without merging** — llama.cpp's `llama-server`
  takes `--lora adapter.gguf` (via `convert_lora_to_gguf.py`). This example merges
  for simplicity; the adapter route lets you hot-swap behaviors.

## Files

| File | Role |
|---|---|
| `build_dataset.py` | seeds + transcripts → `data/{train,valid}.jsonl` |
| `run.py` | full pipeline: dataset → LoRA → fuse → GGUF |
| `config.yml` | serves the fine-tuned GGUF (`kind: llm`) |

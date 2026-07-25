# GLM-5.2 Abliterated + NVFP4 KV — 316K context, 4× DGX Spark

Serving recipe for the **abliterated** (refusal-suppressed) GLM-5.2 with a **true 4-bit NVFP4
KV cache** on a 4-node DGX Spark cluster (GB10, sm_121), vLLM TP=4, MTP k=5.

**316,000-token context · 317,279-token KV pool · 41.4 tok/s peak decode.**

> **What this is.** This serves
> [`0xdfi/GLM-5.2-QuantTrio-Abliterated`](https://huggingface.co/0xdfi/GLM-5.2-QuantTrio-Abliterated)
> — an abliterated derivative of `QuantTrio/GLM-5.2-Int4-Int8Mix`, i.e. a model whose refusal
> behavior has been suppressed. **Provided as-is; it may produce content the base model would
> decline.** Deploy and use responsibly, on hardware you own, in compliance with applicable law
> and the source model's license/gate terms. This repo is a serving/ops recipe only — **no weights,
> no model outputs**.

This combines two of our recipes, both of which you should read first:

- [GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s](https://github.com/tonyd2wild/GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s) — the NVFP4 KV port (400 B/token record, image build, kernels)
- [GLM-5.2-QuantTrio-Abliterated-200K-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-Abliterated-200K-4x-DGX-Spark) — the abliterated weights + shape-swap procedure

The delta here is small: point the **NVFP4 launcher** at the **abliterated weights**. Both are
independent of each other — the NVFP4 record is a KV-cache format, the abliteration is a weight
edit, and they compose cleanly.

## Measured (2026-07-25)

| | fp8_ds_mla | **nvfp4_ds_mla (this recipe)** |
|---|---|---|
| KV record | 656 B/token | **400 B/token (61%)** |
| Max context | 200,000 | **316,000** |
| KV pool | 200,064 | **317,279 (+58.6%)** |
| Peak decode | 41.5 tok/s | **41.4 tok/s** |
| Mean decode (10-prompt suite) | 31.8 | **31.5** |

**Honest headline: NVFP4 buys context, not speed.** Decode is statistically identical to fp8 on
the same k5 config — we measured both back-to-back on the same box, same prompts, same session.
If you see NVFP4 "speedups" quoted elsewhere (including in earlier drafts of our own notes),
check whether the fp8 baseline was running the tuned config; ours wasn't, and that gap was the
tuning, not the KV format.

### Decode by content type (10-prompt suite, warmed, thinking off, temp 0.3)

| group | mean | peak |
|---|---:|---:|
| structured (counting, JSON) | 41.3 | 41.4 |
| code (html, sql, bst, regex, quicksort) | 33.9 | 39.2 |
| prose (explanation, story, reasoning) | 20.9 | 26.4 |
| **all 10** | **31.5** | **41.4** |

**Decode speed is content-dependent and the spread is ~3×.** MTP drafts ~5 tokens per step and
only *accepted* tokens are free, so predictable output runs fast and creative prose runs slow.
Always quote peak **and** mean with the content type — a single number is misleading.

### Why speed dips mid-response (this is not a fault)

Instrumented single run (800 tokens, step rate vs acceptance):

| segment | tok/s | **chunks/s** | tokens/chunk |
|---|---:|---:|---:|
| 0-20% | 36.7 | 7.8 | 4.73 |
| 40-60% | 42.0 | 7.5 | 5.57 |
| 80-100% | 26.1 | 7.3 | 3.57 |

`chunks/s` (engine steps) is **constant**; only tokens-per-step changes, tracking MTP acceptance
as it falls from 6.00/100% to ~4.3/66% when the model reaches genuinely novel content (fresh
arithmetic, verification). `tok/s = steps/s × acceptance` — the GPU is doing identical work
throughout. Nothing to fix; no setting removes it.

## Recipe

Prerequisite: build the NVFP4 image + overlay per the
[NVFP4 repo](https://github.com/tonyd2wild/GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s)
(image `vllm-node-tf5-glm52-b12x:nvfp4-v1`, overlay dir `/var/tmp/glm-triton-nvfp4`), on **all**
nodes. Download the abliterated weights to the **head node only** (workers read them over the
fabric NFS) per the abliterated repo.

Then derive the launcher from the NVFP4 launcher with a **two-line change**:

```
vllm serve /cache/huggingface/hub/glm52-int4-int8mix   →   .../hub/glm52-abliterated
--served-model-name glm-5.2                            →   --served-model-name glm-5.2-abliterated
```

Everything else stays: `--kv-cache-dtype nvfp4_ds_mla`, `--max-model-len 316000`, MTP k=5,
`FULL` cudagraphs with capture sizes `[6,12,18,24,30,36]`, `fuse_gemm_comms`,
`VLLM_MARLIN_USE_ATOMIC_ADD=1`, TP4.

> **Start from the tuned (k5) launcher, not the base one.** The base launcher lacks four speed
> levers (k=5 MTP, tuned cudagraph capture sizes, `fuse_gemm_comms`, Marlin atomic-add). Deriving
> from it costs ~20% peak decode — we did exactly that and spent hours chasing the discrepancy.

### Launch and verify

```bash
# every worker: NFS must be mounted rw and the weights visible BEFORE launching
sudo mount -t nfs -o vers=3 <HEAD_FABRIC_IP>:/var/tmp/models /var/tmp/models
ls /var/tmp/models/hub/glm52-abliterated/config.json    # must succeed on every worker
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches        # all nodes

cd ~/glm-5.2-gb10 && bash launch-abl-nvfp4.sh
```

Readiness is `GPU KV cache size: 317,279 tokens` → `Application startup complete` — **not** a
`/v1/models` 200. Then run a real generation before declaring success.

```bash
curl -s http://<HEAD_IP>:8210/v1/models          # -> "id":"glm-5.2-abliterated", max_model_len 316000
```

## Rollback

```bash
cd ~/glm-5.2-gb10 && ./launch-abl-k5.sh     # abliterated on fp8 / 200K
cd ~/glm-5.2-gb10 && ./launch-tony.sh       # standard censored GLM
```

## Troubleshooting

The failure modes are identical to the fp8 abliterated recipe — model-id 404s, the worker wedge,
the decode stall, the NFS-drop relaunch failure, and "decode is slow" (content, not a fault). See
[that repo's TROUBLESHOOTING.md](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-Abliterated-200K-4x-DGX-Spark/blob/main/TROUBLESHOOTING.md).

## Credits

- **[huihui-ai](https://huggingface.co/huihui-ai)** — the original abliteration of GLM-5.2
  ([Huihui-GLM-5.2-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-GLM-5.2-abliterated-GGUF)).
  The refusal-suppression work everything here derives from is theirs.
- **[0xdfi](https://huggingface.co/0xdfi/GLM-5.2-QuantTrio-Abliterated)** — transplanted that
  abliteration into the QuantTrio format and published the full weights; also the first GLM-5.2
  NVFP4 deployment that set our target numbers.
- **[danielwoz](https://github.com/danielwoz/vllm-dspark-nvfp4)** — the public `nvfp4_ds_mla`
  patch series (E2M1 store kernel, UE8M0 quant scheme) our KV port adapts. Apache-2.0.
- **Keys / [drowzeys](https://github.com/drowzeys)** — the NVFP4-KV image lineage we validated against.
- **b12x**, **[CosmicRaisins](https://github.com/CosmicRaisins/glm-5.2-gb10)**, **jasl** — the
  sm_121 sparse-MLA kernels and CuTe helpers.
- **[QuantTrio](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)** — the base checkpoint.
- Zatz, back199640, ciprianveg, eugr — the 4-Spark GLM recipe lineage.

See [NOTICE](NOTICE) for third-party attribution.

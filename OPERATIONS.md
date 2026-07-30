# Operations notes — verified 2026-07-30

Findings from a full audit and re-characterisation of this deployment on 4× DGX Spark (GB10).
Everything here was verified on the running cluster, not inferred. Where something is unproven
it says so.

---

## 1. `fuse_gemm_comms` is a dead lever on GB10 — it has never been active

The recipe lists `fuse_gemm_comms: true` as one of four speed levers. **It silently resolves to
`False` on this hardware and always has**, including for the published 41.4 tok/s run.

vLLM prints the requested args and the resolved config separately, and they disagree:

```
requested (api_utils.py)  'pass_config': {'fuse_gemm_comms': True}
resolved  (core.py)       'pass_config': {..., 'enable_sp': False, 'fuse_gemm_comms': False, ...}
```

Root cause, in `vllm/config/vllm.py`:

```python
pass_config.sp_min_token_num = get_sequence_parallelism_threshold(hidden_size, tp_size, element_size)
if pass_config.sp_min_token_num is None:
    logger.warning("Model hidden_size too small for the SP threshold heuristic, disabling. ...")
    pass_config.enable_sp = False
    pass_config.fuse_gemm_comms = False
```

Async-TP is built on sequence parallelism, so disabling SP disables `fuse_gemm_comms` with it.
Evaluated live inside the container: `get_sequence_parallelism_threshold(6144, 4, 2)` → `None`.

**The warning text is misleading.** `hidden_size` 6144 is not the problem. In
`sequence_parallelism.py`, device capability **12.1 (GB10) has no entry** in `SP_MIN_HIDDEN_SIZE`
/ `SP_MIN_PER_GPU_SIZE_MB`, so the function returns `None` before comparing any sizes. The lever
is unreachable on this hardware unless `pass_config.sp_min_token_num` is set explicitly.

**Check it yourself:**

```bash
docker logs <container> 2>&1 | grep -oE "'fuse_gemm_comms': (True|False)"
# two hits: the first is what you asked for, the second is what you got
```

### Tested 2026-07-30: forcing it does not work either

The obvious idea is to set `sp_min_token_num` explicitly and bypass the `None`. **That was
tried and it fails.** Launched with:

```
--compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[6,12,18,24,30,36],
  "pass_config":{"fuse_gemm_comms":true,"enable_sp":true,"sp_min_token_num":2048}}'
```

Verified present in the running container, weights loaded to 100%, then the engine died during
worker initialisation:

```
RuntimeError: Worker failed with error 'Failed to send fd: No such file or directory'
RuntimeError: Engine core initialization failed.
```

So on this stack `fuse_gemm_comms` is not merely inert — it is **unreachable**. vLLM's guard is
load-bearing, and pushing past it breaks multiproc worker startup rather than enabling async-TP.

**Practical consequence:** treat `fuse_gemm_comms: true` in this recipe as decorative. It costs
nothing to leave in place (vLLM disables it cleanly) but it is not contributing to the published
throughput, and there is no supported way to turn it on for GB10 without upstream adding a
capability-12.1 entry to `SP_MIN_HIDDEN_SIZE`.

---

## 2. Measured baseline, warm (2026-07-30)

Same weights, same recipe, on a warm engine. Non-streaming (`usage.completion_tokens` is
authoritative), 900–1400-token generations, heavy warm-up first.

| prompt | tok/s |
| --- | ---: |
| sql-bulk (60 templated INSERTs) | **40.0** |
| count300 | 39.3 |
| json60 | 38.5 |
| mult12 | 38.1 |
| prose (250-word explanation) | 24.8 |
| **peak / mean** | **40.0 / 36.2** |

Within 3.4% of the published 41.4 — no regression. The remaining difference is consistent with
MTP acceptance (live metrics show acceptance length swinging 2.50–5.58 against ~5.93–6.0 for the
published peak) and with abliteration perturbing the layer-78 nextn weights the drafter uses.

---

## 3. The bottleneck is comm/weights, not attention

Decode step rate measured against context depth, output held identical so MTP acceptance stays
constant:

| context depth | engine steps/s |
| ---: | ---: |
| 2,000 | 7.6 |
| 32,000 | 7.3 |
| 100,000 | 7.1 |
| 250,000 | 6.7 |

**12% change across a 125× increase in context.** The ~133 ms step is spent on weights and the
4-way all-reduce, not on attention over the KV cache.

Consequences: reducing context or sharding KV for decode (DCP) will **not** buy speed. Levers
that touch the collective — NCCL channel count, async-TP (§1) — are the only ones aimed at the
real cost. `NCCL_MIN/MAX_NCHANNELS` currently ships pinned to `4` on 200 Gb/s links; raising it
is untested here.

*Method warning:* measure decode **after** first-token latency. Computing
`completion_tokens / total_request_time` includes prefill and produces a fake "66% slowdown with
depth" at 32K. Also, with MTP the server emits **one SSE chunk per engine step** (up to k+1
tokens), so counting stream chunks undercounts tokens ~5×.

---

## 4. Cold-start penalty ~30%, and it returns after idle

Immediately after `Application startup complete` — graphs captured, server answering — decode
measured **~30% below** steady state. It recovers after a few long generations.

**It is not only a boot effect.** After ~30 minutes idle, the same container measured
**60.4 tok/s** on a prompt that had been running at **83.5** (observed on the sibling DS4
deployment; same fleet, same behaviour class). Heavy warm-up restored it.

- Benchmark only after several **hundred-token** generations. Short warm-up calls do not clear it.
- Expect the first response after a quiet period to be slow.
- If you are comparing configs, warm both identically or the comparison is meaningless.

---

## 5. `max_tokens` too low returns an empty response

GLM emits a reasoning block first. If the budget runs out before that block closes, the parser
has nothing to hand back:

```
max_tokens 300  → finish_reason: length, content: None, reasoning_content: None, 300 tokens billed
max_tokens 2500 → finish_reason: stop,  content: "<the actual answer>"
```

This looks exactly like a soft-failure/garble symptom and is **not** one. Give agents generous
`max_tokens` on this model. A 700-token answer can need 2000+ of budget.

---

## 6. The NFS weight share is the fragile part of this recipe

Workers read the weights from the head over NFS. When that mount drops — including across a
reboot — the failure is **silent at the worker and misleading at the head**:

```
worker:  huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' ...
head:    torch.distributed.DistStoreError: Timed out after 601 seconds waiting for clients. 1/4 clients joined.
```

The worker cannot see the path, HF falls back to treating it as a repo id, the worker crash-loops,
and the head sits for 10 minutes before failing. Diagnosing from the head alone is a dead end.

**Always verify before launching, on every worker:**

```bash
ls /var/tmp/models/hub/glm52-abliterated/config.json    # must succeed on EVERY node
# if not:
sudo mount -t nfs -o vers=3 <HEAD_FABRIC_IP>:/var/tmp/models /var/tmp/models
```

After any node reboot the mounts are gone and must be re-established.

**Memory note.** GB10 has **unified memory** (121 GB shared CPU/GPU). At
`gpu-memory-utilization 0.91` roughly 110 GB is reserved, leaving ~11 GB for the OS and page
cache. The head is *also* the NFS server, so during a load it reads ~378 GB locally while serving
~378 GB to the workers, all competing for that ~11 GB. This is why the launcher runs
`drop_caches`. **Do not restart the cluster repeatedly in quick succession** — let it settle
between cycles and confirm free memory before relaunching. Consider a lower `gpu-memory-utilization`
on the head than on the workers, or local weight copies per node to remove NFS from the path.

---

## 7. Verifying the abliterated weights are actually loaded

The base `glm52-int4-int8mix` and the abliterated weights are the same size with the same shard
layout, so a path check alone is weak. Two stronger checks:

**Binary.** The edit is confined to `o_proj` in layers 13–77. Compare shards against the base:

| shard containing | expected |
| --- | --- |
| layer 5 `o_proj` (below range) | identical |
| layers 20 / 40 / 77 `o_proj` | differ |
| experts only, no `o_proj` | identical |

**Behavioural.** Ask something a stock GLM deflects. With adequate `max_tokens` (see §5) the
abliterated model answers directly.

Note `config.json` still carries `"name_or_path": "tclf90/GLM-5.2-Int4-Int8Mix"`, inherited from
the base checkpoint and never rewritten. **Do not use it to judge provenance.**

---

## 8. Fabric health check (all verified good here)

`NCCL_DEBUG=WARN` hides the transport line, so confirm RDMA is really being used via port
counters rather than logs:

```bash
cat /sys/class/infiniband/rocep1s0f0/ports/1/counters/port_xmit_data
```

Non-zero and growing, symmetric across nodes, means RDMA. A TCP fallback leaves these at zero.
Measured here: ~14 TB per node, balanced across all four — 200 Gb/s HDR links, MTU 9000, no
errors, RTT ~0.5 ms, no thermal or clock throttling.

Also noted: a **second 200 Gb/s RoCE port is ACTIVE but unused** on every node (link-local
address, zero RDMA traffic). A second fabric path exists and is not in the mesh.

# vast-spot-toolkit

![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Zero telemetry](https://img.shields.io/badge/telemetry-zero-blue)

Host filter for Vast.ai + background checkpoint watcher for R2/S3. MIT licensed, no dependencies beyond
`curl`/`jq` and `aws-cli`.

1. **`vast-host-filter.json`** — host-filtering query for 7B–70B runs, with the reasoning for every field.
   Cuts out hosts that look cheap per hour but 4× your bill on traffic or slow PCIe.
2. **`ckpt-watcher.sh`** — background uploader that only syncs checkpoints marked complete, so a mid-save
   crash never corrupts remote storage.

---

## Why these exist

Community GPU marketplaces rent an RTX 4090 for well under a dollar an hour. Two lessons cost us real money:

- **The cheapest host is rarely cheap.** On one QLoRA run (Qwen2.5-7B, 300 steps) the GPU hour cost $0.13, but the
  host billed **$0.66 of traffic** for the 15 GB model download — the run cost $0.81 instead of ~$0.19 (single
  measured run; your numbers will differ, the trap is structural). Different 4090 hosts also ran the same script
  at 4 s/step vs 6 s/step.
- **Spot hosts die mid-run.** Reboots, network drops, outbid instances. If your checkpoint only exists on the
  machine, the run is gone.

## Tool 1: the host filter

```bash
curl -s -X POST https://console.vast.ai/api/v0/bundles/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VAST_API_KEY" \
  --data @vast-host-filter.json | jq '.offers | length'
```

What each field does and why:

| Field | Value | Why |
|---|---|---|
| `verified` | `true` | Unverified hosts are a lottery on top of a lottery. |
| `reliability2` | `≥ 0.98` | Vast's own reliability score over a long window. For multi-day runs we raise it to 0.99. |
| `inet_down` | `≥ 500` Mbps | A 15–30 GB image+model re-downloads on every restart; slow hosts bill you for waiting. |
| `inet_up` | `≥ 100` Mbps | Checkpoints upload off-machine every few minutes. |
| `inet_down_cost`, `inet_up_cost` | `≤ $0.01`/GB | **The $0.81 lesson.** Per-GB traffic prices differ by 100× between hosts ($0.0003–0.04). This single filter removes the traffic trap. |
| `cuda_max_good` | `≥ 12.1` | Recent PyTorch images need it. |
| `disk_space` / `allocated_storage` | `≥ 60` GB | Image + model + dataset + checkpoints. |
| `type` | `on-demand` | Use `bid` only if your job resumes cleanly from checkpoints — every preemption is a re-download. |

**Two fields the API can't filter on — check them client-side:**

```bash
... | jq '[.offers[] | select(.pcie_bw >= 10 and .disk_bw >= 1000)]'
```

- `pcie_bw ≥ 10` GB/s — PCIe 4.0 x16 is ~25 GB/s; mining-rig x1/x4 risers read ~3 GB/s and throttle training.
- `disk_bw ≥ 1000` MB/s — slow disks stretch every checkpoint save and every restart.

**Sort by all-in price, not `dph_total`.** The GPU hour is not the real cost. We rank by:

```
all_in = dph_total
       + expected_download_gb × inet_down_cost / expected_hours   # one-time pull, per hour of the run
       + expected_upload_gb_per_h × inet_up_cost                  # checkpoint traffic
```

For a 4-hour 7B run: `expected_download_gb ≈ 20` (image + model + dataset), `expected_upload_gb_per_h ≈ 2`.

**Adapting for model size:** for 7B QLoRA a single RTX 4090 (24 GB) works; for 13B use `"gpu_name": {"in":
["RTX 4090", "RTX A6000"]}`; for 70B QLoRA switch to 80 GB cards (`"gpu_name": {"eq": "A100 SXM4"}` or H100) and
raise `gpu_ram` accordingly. The rest of the filter stays the same.

## Tool 2: the checkpoint watcher

`ckpt-watcher.sh` syncs checkpoints to S3/R2 — but only ones your training script has marked as **completely
written**. Uploading from inside the training loop is how checkpoints get corrupted: a segfault or hang
mid-upload leaves a truncated file in storage, and you resume from garbage.

**Contract** (fits any `checkpoint-<step>/` layout — HF Trainer, Axolotl, Unsloth, LLaMA-Factory):

```
ckpt/
  checkpoint-100/         complete, contains .done  -> uploaded
  tmp-checkpoint-150/     still being written       -> skipped
  checkpoint-150/         complete, contains .done  -> uploaded
```

Touch `.done` right after each checkpoint finishes writing. For HF Trainer / Unsloth, drop in a 5-line callback:

```python
from pathlib import Path
from transformers import TrainerCallback

class MarkCheckpointDone(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        (Path(args.output_dir) / f"checkpoint-{state.global_step}" / ".done").touch()
```

Run it next to your training (same container/host):

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
export S3_URI=s3://my-bucket/my-job
export S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com   # R2; omit for AWS S3
./ckpt-watcher.sh          # SIGTERM triggers one final sync before exit
```

Tunables: `CKPT_DIR` (default `/workspace/ckpt`), `READY_MARKER` (default `.done`), `SYNC_INTERVAL_S`
(default `120`). Use a credentials pair scoped to just your bucket/prefix — this file ends up on a stranger's
machine, that's the whole point of spot.

## Scope & Limitations (DIY vs. Automated Failover)

This toolkit runs locally / inside the GPU container. It solves bad host selection and off-machine backups,
but **a script inside a dead container cannot replace its own machine**.

| Failure mode | `vast-spot-toolkit` (DIY) | External Supervisor ([NomadGPU](https://nomadgpu.com)) |
|---|---|---|
| High bandwidth fees / slow PCIe hosts | Filtered before renting | Filtered before renting |
| Checkpoint lost when host dies | Synced to R2/S3 via watcher | Synced via scoped per-job temporary R2 keys |
| Host reboots or disappears mid-run | Manual: rent new GPU & resume | Auto-rents replacement & resumes (median ~1 min; live recovery rate on [status.nomadgpu.com](https://status.nomadgpu.com/s/status)) |
| Silent GPU hang (0% util, still billing) | Keeps billing until you notice | Auto-kills after 5 min GPU hang / 30 min stall |
| Runaway bill overnight | Manual monitoring | Hard dollar caps per job / day / week / month |

### Need hands-off failover?

This toolkit was extracted from **[NomadGPU](https://nomadgpu.com)** — a managed supervisor that runs on top of
your own Vast.ai account (restricted 5-permission API key; you pay Vast directly).

If you want to verify failover before trusting a long run, `nomadgpu dry-run` (~$0.10–$0.50 in Vast compute)
spins up a host, saves a checkpoint to R2, destroys the instance on purpose, and verifies `resumed=true` on a
fresh GPU.

- **Live recovery telemetry:** [status.nomadgpu.com/s/status](https://status.nomadgpu.com/s/status)
- **Pilot access (14 days free):** send your Axolotl / Unsloth / PyTorch training command to
  [ops@nomadgpu.com](mailto:ops@nomadgpu.com?subject=NomadGPU%20pilot%20config) — we'll wrap it into a resilient
  job spec the same business day. Prefer to start yourself? [Sign up here](https://app.nomadgpu.com/portal/signup) —
  every application is reviewed by hand, usually the same business day.

Issues and PRs welcome — especially host-selection heuristics: what signals do you use to spot a bad 4090 before
renting?

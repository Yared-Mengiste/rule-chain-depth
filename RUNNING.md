# Running the shared 40-question benchmark

The `benchmark` command tests identical worlds and questions with PLN-RAG,
normal RAG, and an LLM given full context. PLN-RAG can use your Docker image;
normal RAG and full context run in the host Python process. The sibling
repositories are read without changing their source or `.env` files.

## What is implemented

| Dataset | Worlds | Questions | Sentences per world | Facts / rules across worlds |
|---|---:|---:|---:|---:|
| Original pilot | 8 | 16 | 12 | 57 / 39 |
| Full benchmark | 20 | 40 | 12 | 150 / 90 |

The full dataset has five supported questions and five unsupported controls in
each depth bucket, 0–3. It has five fact worlds, six simple-chain worlds, three
conjunction worlds, three relational worlds, and three mixed worlds. Depths are
interleaved, so `--world-limit 4` selects one world at each depth.

Each supported/control pair shares one world. Facts and rules are counted as
entries, not unique sentences across the dataset. This is a small controlled
synthetic benchmark with repeated structural templates, not a broad language
reasoning evaluation.

The full fixtures live in [data/benchmark/worlds.json](data/benchmark/worlds.json)
and [data/benchmark/hand_expected.json](data/benchmark/hand_expected.json).
[b1/dataset.py](b1/dataset.py) authors the cases and explicit proof annotations.
Both independent solvers validate every answer, exact depth, supplied proof,
and unsupported control's missing-fact repair before any model call.

## Configuration

All shared-run settings come from **this folder's `.env`**, with inherited
environment values taking precedence. The runner does not load `../normal-rag/.env`.

For the same free Nvidia route used in your pilot:

```dotenv
OPENAI_API_KEY=your-openrouter-key
OPENAI_MODEL=openrouter/nvidia/nemotron-3-super-120b-a12b:free
OPENROUTER_PROVIDER=nvidia
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
PARSER=canonical_pln
OLLAMA_MODEL=nomic-embed-text
OLLAMA_URL=http://127.0.0.1:11434/api/embeddings
B1_QDRANT_URL=http://127.0.0.1:6333
B1_LLM_TEMPERATURE=0
B1_LLM_MAX_TOKENS=2000
B1_LLM_TIMEOUT=90
B1_QUERY_TIMEOUT=240
B1_LEARN_TIMEOUT=900
B1_REASONING_TIMEOUT=30
B1_EMBEDDING_TIMEOUT=30
B1_MAX_LLM_CALLS=500
B1_LLM_MIN_INTERVAL=4
```

The 500-call cap is an **example**, not an automatic change to your `.env`, a
dollar budget, or an OpenRouter quota. Set a cap appropriate to your account.
Buying credits does not edit local settings or change a `:free` model into its
paid route. Model/provider availability and account limits still apply.

One cap covers the entire comparison, including PLN-RAG sentence parsing,
question parsing, answer generation, and both baselines. The default order is
PLN-RAG, normal RAG, full context. Earlier systems can exhaust the shared cap.
A new run has a fresh local counter; your remote account allowance is shared.

A rough full-run estimate is 300 PLN calls (240 sentence parses, 40 question
parses, 20 answer generations) plus 40 calls per baseline: about **380 calls**
before extra PLN formatting/parsing calls. Some unsupported or missed proofs
avoid answer-generation calls. This estimate is not a guarantee. The runner
keeps your configured cap, including the original 40 if you have not changed it.

The shared runner strips the `openrouter/` routing prefix for the baseline's
direct API request. The model ID, provider pin, API key/base URL, temperature,
output limit, and LLM timeout are otherwise matched. All embeddings use local
Ollama with `nomic-embed-text`. Baseline retrieval remains exactly top four;
full context skips retrieval and uses the identical baseline prompt.

## Validate without model calls

From `rule-chain-depth`:

```bash
python3 -m b1 oracle \
  --worlds data/benchmark/worlds.json \
  --expected data/benchmark/hand_expected.json \
  --output artifacts/oracle/benchmark.answer_key.json

.venv/bin/python -m unittest discover -s tests -v
```

Tests use fake HTTP/model responses and do not measure model accuracy. Full
integration tests require the sibling `../normal-rag` source directory.

Save a baseline launch plan without contacting Ollama or OpenRouter:

```bash
python3 -m b1 benchmark --systems rag full_context --prepare-only
```

Prepare the complete comparison, including inspection of your Docker image:

```bash
sudo python3 -m b1 benchmark \
  --container pln-rag-pln-rag-1 \
  --prepare-only
```

`--prepare-only` validates data/configuration and saves a new plan. It does not
run dependency smoke checks, check live model availability, or produce scores.
The later live invocation creates its own new run; it does not resume the plan.

## Run PLN-RAG in Docker and both baselines locally

```bash
sudo python3 -m b1 benchmark \
  --container pln-rag-pln-rag-1
```

This selects all 20 worlds: **40 questions per system, 120 records total**.
The host command needs Python 3.10+ and the sibling normal-RAG source; its
baseline uses only the standard library. Docker supplies PLN-RAG's Python and
native dependencies. Ollama and Qdrant must be reachable at the configured URLs.

For a smaller trial through all three systems:

```bash
sudo python3 -m b1 benchmark \
  --container pln-rag-pln-rag-1 \
  --world-limit 4
```

That produces eight questions per system, 24 records total. Use `--world-limit 1`
for the smallest two-question-per-system trial.

To use the original pilot data on all systems, add `--dataset pilot`. This
selects its eight worlds and 16 questions unless a smaller world limit is given.
The older `pilot` and `docker-pilot` commands still run only PLN-RAG's original
pilot and retain their existing options.

## Select individual systems

```bash
# PLN-RAG only, using Docker: 40 questions
sudo python3 -m b1 benchmark --systems pln_rag \
  --container pln-rag-pln-rag-1

# Normal RAG and full context only, on the host: 80 records
python3 -m b1 benchmark --systems rag full_context

# Normal RAG only: 40 records
python3 -m b1 benchmark --systems rag

# All systems with an installed local PLN runtime
python3 -m b1 benchmark --pln-runtime local \
  --repo ../PLN-RAG --python .venv/bin/python
```

Flags also include `--normal-rag-repo PATH`, `--qdrant-url URL`, `--ollama-url URL`,
`--worlds PATH`, `--expected PATH`, and a unique `--run-id NAME`. Explicit service
URL flags override `.env`. The shared Docker command otherwise honors
`B1_QDRANT_URL` in `.env`; absent that setting, its Docker default is port 6333.

## How Docker receives the same key and settings

The launcher reads this folder's configuration into a subprocess environment.
Docker receives `--env OPENAI_API_KEY` and other **variable names**, with values
supplied in that environment. The saved command contains no credential value.
This explicitly forwards the matched host settings even when the selected image
has its own defaults. The standalone normal-RAG `.env` is not involved.

The launcher starts a separate temporary container from the existing container's
exact image ID, mounts this benchmark folder, and reuses only individually
mounted parser JSON artifacts read-only. It does not mount the live application's
data volume. New world collections and fresh query snapshots isolate the run.
Normal RAG runs in the host process, outside that container.

For the shared `benchmark` command, Docker runs with the host launcher's effective
UID/GID so both processes can write the run's artifacts and request budget.
With `sudo`, both run as root and create root-owned files. Use the command without
`sudo` when your account already has Docker access. The standalone `docker-pilot`
continues to run its workers as the benchmark folder's owner.

Older launchers used the folder owner's UID/GID for the shared benchmark too.
Under `sudo`, that caused `PermissionError` on `docker.doctor.json.tmp` in the
new root-owned run directory. Changing permissions on existing artifacts did not
fix the next run's directory. Rerun with the updated launcher; no recursive
`chmod` is needed.

## Read the results

Each invocation creates `artifacts/benchmark/<run-id>/`:

| File | Contents |
|---|---|
| `manifest.json` | Selected systems/worlds, settings, source hashes, dataset fingerprint, image ID, and run stage. |
| `worlds.json`, `hand_expected.json`, `oracle.json` | Frozen full input dataset and independently validated answers. Selected IDs are in the manifest. |
| `report.md` | Overall and per-depth accuracy, error counts, and links to each record. |
| `summary.json` | Per-depth, supported-only per-depth, per-kind and per-label scores; descriptive Wilson 95% intervals; provider/model reports; token totals and missing-measurement counts. |
| `records.json` | One record per selected question/system, including errors and explicit skipped records. |
| `llm_requests.json` | HTTP audits grouped by system, including provider/model, usage, pacing, requests, and raw response text. |
| `docker.launch.json`, `docker.doctor.json` | Docker setup evidence when using the container runtime. |
| `pln_rag/raw.md` | Learned atoms, proof outputs and worker logs, if the PLN runner started. |
| `<system>/<world>/<question>/record.json` | Individual result, expected label, depth grouping, timing, retrieval/citation diagnostics, and errors. |
| Baseline operation `input.json`, `request.json`, `response.body`, `raw_model_output.txt` | Exact public input, chat request without credentials, unchanged API response bytes and model content. Files exist only when the corresponding operation occurred. |

`COMPLETE` means the selected run finished without error records; it does not mean
every answer was correct. `COMPLETE_WITH_ERRORS` retains errors in the score.
`complete` counts records; `execution_complete` also requires no explicitly
skipped questions. `full_dataset_selected` distinguishes a subset from the full
dataset. Exit status is 0 without errors, 2 with recorded errors, and 1 for setup
failure. Prepared runs contain no `summary.json` or model results.

No error becomes a correct abstention. After a fatal baseline API/transport/budget
failure, remaining questions in that system receive explicit unattempted `ERROR`
records. If PLN stops during learning, missing questions also receive explicit
unattempted records. Other selected systems are still attempted under the same
remaining request budget. Existing wrong answers and raw outputs are not repaired.

`proof_coverage` distinguishes:

- Whether the supplied proof's sentence IDs were retrieved.
- Whether **any** proof exists in the retrieved subset, checked by the oracle
  after the model answered.
- Whether the full/shown context proves the claim.
- How many sentences the supplied proof needs and whether it can fit in four.

No-proof controls and unavailable measurements use `unmeasured`. The answer
key, depth, logical forms, and proof IDs are never passed into model prompts.
Some valid proofs need more than four sentences, so top-four RAG has an explicit
capacity limit; compare it with full context rather than increasing retrieval.

PLN learning time is separate from query time. Query time includes LLM and
pacing waits; normal-RAG index-building time is stored separately per question.
Wilson intervals are descriptive: shared worlds and templates violate the
independence assumption of a simple binomial sample.

The [documented one-world run from 2026-10-07](artifacts/analysis/20261007T111319Z-d88e08.md)
completed with PLN-RAG 0/2 and RAG 2/2. Its evidence explains a malformed PLN
query, a control proved using a fact added from the question, four sentence parse
failures, and model-output truncation. The note preserves the recorded results
and links to the original artifacts.

## Files implementing the expansion

| File | Responsibility |
|---|---|
| [b1/dataset.py](b1/dataset.py) | Authors and admits the balanced full dataset. |
| [b1/comparison.py](b1/comparison.py) | Shared settings, orchestration, post-answer evaluation, error completion and reports. |
| [b1/baseline.py](b1/baseline.py) | Loads the existing baseline without editing it, matches settings and adds shared request accounting. |
| [b1/pln_runner.py](b1/pln_runner.py) | Runs either dataset with fresh PLN workers and an optional shared output/budget location. |
| [b1/docker_runtime.py](b1/docker_runtime.py) | Launches the shared run's PLN component from the inspected image with matched environment values. |
| [tests/test_comparison.py](tests/test_comparison.py) | Checks the new dataset, complete orchestration, input isolation, budgets, raw outputs and Docker launch contract offline. |

To intentionally regenerate full fixtures from their authored source:

```bash
python3 -m b1 author-benchmark
```

This overwrites the full fixture files after oracle validation. It does not modify
the original pilot fixtures or invoke an LLM.

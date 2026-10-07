# B1 — rule-chain depth

For a detailed explanation of the logic, Docker/API-key flow, every project
file, configuration settings, and result interpretation, read [GUIDE.md](GUIDE.md).

A separate benchmark project for PLN-RAG, sentence retrieval with an LLM, and
an LLM given the full world. The work proceeds through a required pilot gate.
No accuracy results are supplied without a live run.

The original eight-world pilot and the balanced **40-question dataset** are
implemented. The shared runner tests PLN-RAG, normal RAG and full context on
identical inputs. PLN-RAG can run in Docker while both baselines run on the host.
See [RUNNING.md](RUNNING.md) for configuration, request budgets and result fields.

```sh
# Validate and inspect without model calls:
sudo python3 -m b1 benchmark --container pln-rag-pln-rag-1 --prepare-only
# Live comparison: 40 questions per system, 120 records:
sudo python3 -m b1 benchmark --container pln-rag-pln-rag-1
```

The existing `.env` request cap remains in force. A full three-system run is
roughly 380 chat calls before additional PLN parsing calls; the old 40-call cap
will not cover it. Choose a budget compatible with your account before a live
run. Implementation/offline tests are separate from live performance results.

## Run the oracle first

Python 3.10 or later is sufficient; the oracle has no third-party dependencies.

```sh
python3 -m unittest discover -s tests -v
python3 -m b1 oracle
```

[The frozen pilot worlds](data/pilot/worlds.json) contain exactly twelve shuffled
sentences each. [The hand annotations](data/pilot/hand_expected.json) were authored
independently of the solvers. [The generated answer key](artifacts/oracle/pilot.answer_key.json)
records labels, exact depths, sentence IDs, and alternative proofs.

The first four worlds test depths 0, 1, 2, and 3. The next four test conjunction
at depths 1 and 3, a relational hop at depth 1, and a mixed proof at depth 2.
Each has a supported target and an unsupported control: **8 worlds, 16 pilot
questions**, separate from the implemented 40-question experiment.

To reproduce the frozen fixtures from their hand-authored source:

```sh
python3 -m b1 author-pilot
```

Both solvers must agree before any dataset is admitted or any system is called.
A disagreement, shortcut, incorrect unsupported label, changed English sentence,
or incorrect hand annotation stops execution. The forward solver joins known
facts and saturates the world. The backward solver matches the goal and searches
its premises, with independent substitution and cycle handling.

## Configure and run the PLN-RAG pilot

The default configuration is `canonical_pln` with `openai/gpt-4o-mini`.
Create `.env` using [.env.example](.env.example) and configure `OPENAI_API_KEY`
there. Never put credentials in a dataset or result file.

PLN-RAG is imported from `../PLN-RAG` without editing its source. The live
environment additionally needs DSPy, NL2PLN, PeTTaChainer, PeTTa, the Janus bridge,
SWI-Prolog, Qdrant, and Ollama with `nomic-embed-text`. The setup checker calls no
LLM and labels its output explicitly as setup information, not benchmark data.

```sh
python3 -m b1 doctor
python3 -m b1 pilot
```

Use `--repo /path/to/PLN-RAG` and `--python /path/to/venv/bin/python` for another
configured runtime. Local rootless dependencies live under `runtime/`; installed
Python packages live in `.venv/`. The runner can start a private Qdrant binary at
`runtime/bin/qdrant`, listening on port 16333. Alternatively set `B1_QDRANT_URL`
to an existing service. It creates a new collection for each world and run.
It never resets an existing application's memory or collection.

Each run writes `artifacts/pilot/<run-id>/raw.md`, per-operation JSON and logs,
`records.json`, `learning.json`, the oracle, source hashes, and `gate.json`.
These expose every English input, parsed atom, query, proof, error, and timing.
Credentials are redacted from outputs. A failed capability pilot requires user
review before building baselines or scaling. An infrastructure failure stops the
pilot and is not evidence that a rule kind is unsupported.

The local Dockerfile's PeTTaChainer URL/pin was unavailable during setup; the
runtime setup record must disclose any resolved replacement dependency revision.
Do not describe such a run as reproducing the unavailable pinned runtime.

## PLN-RAG-only trial using the existing Docker image

The standalone normal-RAG project is not called by this command. Configure this
folder's `.env` with the OpenRouter key under `OPENAI_API_KEY`:

```dotenv
OPENAI_API_KEY=your-openrouter-key
OPENAI_MODEL=openrouter/nvidia/nemotron-3-super-120b-a12b:free
OPENROUTER_PROVIDER=nvidia
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
PARSER=canonical_pln
B1_LLM_TEMPERATURE=0
B1_LLM_MAX_TOKENS=2000
B1_MAX_LLM_CALLS=40
B1_LLM_MIN_INTERVAL=4
```

From this directory, run with a user that has Docker access:

```sh
sudo python3 -m b1 docker-pilot --container pln-rag-pln-rag-1 --world-limit 2
```

This runs the first two frozen worlds (depths 0 and 1), four questions total.
Both oracles validate all eight worlds before the subset is selected. A clean
two-world trial is `TRIAL_COMPLETE`, never a full-pilot pass. Increase
`--world-limit` to 8 only when deliberately running the complete pilot and when
the account has sufficient remaining requests.

The launcher inspects the running container and starts a separate container
from its exact image ID, without rebuilding or editing PLN-RAG. It mounts only
the benchmark folder and the live container's individual tuned parser JSON
files (read-only). It does not mount the live application data volume or call
the running application's ingest, query, or reset endpoints. New collections
have unique run/world names; question workers still restore immutable snapshots.
The temporary container is removed after the run; benchmark artifacts persist.
The image digest is recorded. Local `runtime/deps` and SWI-Prolog overrides are
disabled inside Docker so the image's own dependencies are used.

The Linux launcher uses host networking to reach the existing Qdrant service at
`http://127.0.0.1:6333` and Ollama at
`http://127.0.0.1:11434/api/embeddings`. Override these with `--qdrant-url` and
`--ollama-url` if needed. `--prepare-only` inspects the source container and saves
the exact launch plan without running it. A dependency smoke test runs inside
the image before any LLM call.

Requests pin the named OpenRouter provider and disable routing fallback and
automatic LLM retries. The request budget and spacing are shared across all
world-learning and query workers, including additional requests caused by
upstream DSPy/parser formatting fallbacks. An HTTP failure prevents further
model requests in that worker. Learning failures stop the trial. The 40-request
cap applies to this run only; other uses of the same account share its quota.

`artifacts/pilot/<run-id>/raw.md` shows learned atoms, answers, and proofs.
`records.json` contains per-question results, and `llm_requests.json` contains
the observed HTTP request bodies, untouched response text, reported model and
provider, token usage, request latency, and pacing wait. Authorization headers
are excluded. Missing token/provider fields are `unmeasured`, not zero or guessed.
The final gate records the actual number of requests sent. An interrupted worker
may have sent requests whose response records were lost; the shared budget file
still counts those attempts.

Answer time includes pacing waits as experienced by the service. Each request
also records `rate_wait_seconds` separately so those waits can be distinguished
from API execution time. The default 2,000-token limit applies to both parsing
and proof-to-answer calls; model reasoning tokens can consume that limit.

## Fairness and diagnostics

Every question runs in a fresh process. The world is learned once, saved to an
immutable atom file, and copied into a new work directory before each question.
Hashes verify the copy; before/after hashes and query-added statements expose
mutation. Vector collections contain only that world's learned sentences;
queries retrieve from them without adding vectors. Background knowledge is off.

The same LLM configuration is used for parsing and proof-to-answer generation:
temperature 0, output limit 2000, cache off, automatic retries off. Answer time
includes the normal query path and answer generation. Restore time, process
setup time, total worker time, world learning time, and per-sentence learning
time are recorded separately. A hard process-group deadline also kills nested
reasoning workers.

PLN-RAG's label comes from the proof, never from its answer wording. Observers
record engine/LLM exceptions before upstream code can swallow them. A timeout,
crash, malformed query, or absent query is `ERROR`, never a correct abstention.
Sentence parse failures count rejected, partially rejected, or empty ingestion
results, once per sentence. A syntactically accepted but semantically wrong parse
still requires inspection of the raw atoms; acceptance is not a correctness test.

Proof sources are resolved by exact atom-name tokens, not approximate semantic
search. Colliding atom names are marked ambiguous rather than guessed. These
diagnostics do not replace a system's proof label with the oracle's label.

## Depth and the 40-question experiment

A fact has depth 0. A rule application has depth `1 + max(premise depths)`;
the oracle minimizes that value over proofs. Two facts in a relational hop
therefore yield depth **1**, not 2. A mixed hop followed by a simple rule has
depth at least 2. Depth-0 questions are labelled `fact`, since their proof uses
no rule kind.

An unsupported question has **no finite proof depth**. Its `depth` is `null`.
Its `depth_bucket` describes the paired supported target and is independently
checked by adding one missing fact: the repaired control must have that exact
depth. Thus bucketed accuracy must not be described as the actual proof depth
of unsupported questions.

The full experiment has five supported questions and five unsupported controls
in each depth bucket: 20 worlds and 40 questions. Its 240 sentence entries contain
150 facts and 90 rules. The `benchmark` command produces bucketed and
supported-only per-depth accuracy with descriptive Wilson 95% intervals. Paired
questions and repeated templates limit generalization and violate independence.

Normal RAG retrieves exactly four sentences; full context receives all twelve.
Both reuse the sibling baseline's existing prompt and answer contract. Unknown
citations are dropped and counted while raw outputs remain untouched. After each
answer the benchmark records supplied-proof coverage and independently checks
whether any proof exists in the retrieved subset. Neither diagnostic enters a
model prompt. Some proofs need more than four sentences; that limitation is
reported explicitly rather than changing retrieval.

Reports separately expose accuracy, parse-failure share, errors/timeouts,
retrieval sufficiency, full-context outcomes, and learning/answer times. See
[RUNNING.md](RUNNING.md) for the shared runner and [GUIDE.md](GUIDE.md) for the
underlying logic and original pilot walkthrough.

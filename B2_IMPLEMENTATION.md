# Fixed B2 implementation and offline verification

Implemented on branch `feature/b2-fixed-entailmentbank-20261016`, created from
the actual local `main` commit
`05a56eccccaed37b783b42bfc3384f87456b664f` on 16 October 2026.
The implementation stays local and unmerged. Existing branch pointers and
historical artifacts are preserved. The pre-existing untracked specification
and `benchmark_explained.pptx` are outside the implementation commit.

B2 adds its own package/entry point, checked-in dataset, single comparison
runner, local evidence/resource reports, isolated workers, and documentation.
B1 source, synthetic data, oracle, and CLI behavior are unchanged; B2 reuses
its environment, baseline loader, snapshot, and request-control helpers.
No sibling repository source was edited.

The fixed bundle retains 42 selected original examples, six at each calculated
annotated depth 1–7, and all 25 sentences per example. Selection uses one
`random.Random(17)` and the prescribed source-order group shuffles, interleaved
in six rounds. Exact IDs and source-wide discrepancies are in `data/b2/source.json`.
Selected `Mercury_400025` retains original depth 5 and calculated depth 4.

| Bundle file | SHA-256 |
|---|---|
| `worlds.json` | `eccce84741276ab9bee2a204593210472e3977159d89218925e137c3d85187a0` |
| `references.json` | `49c98373b4377023e75ab267b989108e69a4a83ff4c5a8f96c8e25d3dc4ab270` |
| `source.json` (bundle identity) | `d834aa3a22d99dd26d4928d9c97c62b5c75ac0aaf86ca08cc498aba3ca3a912b` |
| `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

Offline checks completed:

- `python3 -m unittest discover -s tests -v`: **82 tests passed**, including all
  B1 regressions, bundle integrity/order, annotation isolation, native query
  mutation guards and swallowed timeouts, evidence availability, a mocked
  126-record comparison, matched settings, shared caps, and real process deadlines.
- `python3 -m b2 validate`: 42 unique worlds, six at every calculated depth;
  selected source-depth discrepancy reported explicitly.
- `python3 -m b2 benchmark --prepare-only`: 42 worlds, all three systems,
  126 planned records, initial estimate 1,176 calls; the configured local cap
  remained 500. Tests also verify 56 calls for two worlds/all systems and two
  calls for two worlds/full context.
- Clean copied-checkout validation/preparation succeeded with only B1/B2 code
  and the bundled dataset, no credentials or sibling sources, and sockets
  explicitly disabled in the subprocesses.
- Reauthoring from the checksum-verified local pinned source/license reproduced
  all four machine/license files byte-for-byte.
- CLI help and `git diff --check` passed; no B1 source/data differences from the
  resolved `main` base.

No live model or embedding calls were made, and no live performance is claimed.
The native adapter was checked against the sibling interfaces and exercised
with mocked dependencies, but a real image/parser/Qdrant/Ollama run remains
unverified. Runtime compatibility, native proof syntax coverage, and semantic
parse quality must be assessed from a later live run. Unrecognized or ambiguous
proof sources stay unavailable. Local workers use public staged inputs but are
not a filesystem sandbox; Docker uses narrow mounts. The native chaining-step
setting is recorded without claiming enforcement; process deadlines apply.

After configuring credentials, service locations, and a deliberate request
budget, the complete live command is:

```sh
python3 -m b2 benchmark --container pln-rag-pln-rag-1
```

Use `--world-limit 2` for the same fixed dataset's two-world trial. The initial
full estimate exceeds a 500-call cap, and parser fallbacks can add requests;
the runner does not change the cap automatically.

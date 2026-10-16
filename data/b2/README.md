# Fixed EntailmentBank Task 2 bundle

This bundle contains **42 worlds**, six at each annotated graph depth 1–7,
with **25 original sentences** and one original hypothesis per world. All
distractors are retained. A full PLN-RAG/RAG/full-context comparison has 126
records. Full context uses the same RAG explanation prompt with all sentences.
Normal validation and execution use only this bundle; no training, download,
or split selection is required.

## Attribution and source

AllenAI EntailmentBank, Dalvi et al. (2021),
[Explaining Answers with Entailment Trees](https://aclanthology.org/2021.emnlp-main.585/).
The upstream [Apache 2.0 license](LICENSE) is preserved byte-for-byte.

- Repository: <https://github.com/allenai/entailment_bank>
- Revision: `daac2fdb7ab52ec3ef8f2953f59288c1edd7c2f0`
- Original development source: `data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset/task_2/dev.jsonl`
- Source SHA-256: `3271adc67c65149780adbd3729f6b19404ff288e1849905fc16c1c22814a28f7`
- License SHA-256: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`

The source filename is provenance, not a user-facing benchmark mode. Only the
42 selected original records are retained here. Full upstream files and
authoring caches are neither tracked dependencies nor runtime prerequisites.

## Deterministic selection

Structural validation checks sentence/intermediate IDs and text, multi-premise
and branching steps, missing references, cycles, root reachability, and the
hypothesis. Input sentences have depth 0; each conclusion has depth
`1 + max(premise depths)`. The hypothesis depth supplies the group. This checks
the annotated graph, not logical validity or a minimum proof depth.

The pinned source has 51, 64, 28, 18, 13, 7, and 6 examples at depths 1–7.
One `random.Random(17)` shuffles copies of those groups, preserving source order
before shuffling, in ascending depth order. The first six in each group are
selected. Output interleaves six rounds of depths 1–7; the first two worlds
cover depths 1 and 2. Selection never uses system performance.

[source.json](source.json) records the exact source-order inventories, selected
IDs/order, recipe, depths, revision, checksums, and source discrepancies. Display
IDs `b2-001` through `b2-042` map reversibly to source IDs in that file and
[references.json](references.json). Public [worlds.json](worlds.json) contains
only display IDs, hypotheses, and original sentence IDs/text.

All selected original records and annotations are preserved in the evaluator
references. In selected world `b2-039`, source `Mercury_400025` has original
`depth_of_proof=5` but calculated graph depth 4. Both values are retained; it
belongs to the calculated depth-4 group. `length_of_proof` is likewise preserved
and checked. Source-wide depth discrepancies are recorded without rewriting
the annotations.

## Maintenance reproduction

Obtain the pinned source and license once, or use a verified existing local
copy. The authoring command verifies both checksums before reading annotations:

```sh
python3 -m b2.author /path/to/dev.jsonl /path/to/upstream/LICENSE /tmp/b2-reproduced
python3 -m b2 validate
```

The first command reproduces the four machine/license files deterministically;
this explanatory README is maintained separately. Compare `worlds.json`,
`references.json`, `source.json`, and `LICENSE` byte-for-byte with this directory.
The bundle identifier is the SHA-256 of `source.json`, which contains checksums
for the other three files. Validation checks integrity, public/reference
agreement, selection order, and graph metadata without needing upstream data.

Workers receive only public inputs, never the source metadata, proof, leaves,
intermediates, distractor labels, or depth. Scores are local annotated-evidence
overlap, not official proof accuracy. This balanced subset does not represent
the full EntailmentBank distribution. See [B2.md](../../B2.md) for commands,
configured request budgets, and evidence/resource interpretation.

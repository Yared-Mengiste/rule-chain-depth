# B1: 2-world PLN-RAG trial

Live outputs only. This pilot is separate from the planned 40-question benchmark.

The label comes from the proof payload, with runtime/translation errors taking precedence.

Gate: **REVIEW_REQUIRED**. No baselines or 40-question results are included.

| World | Kind | Supported depth | Target label | Control label | Parse failures |
|---|---|---:|---|---|---:|
| pilot-01 | fact | 0 | SUPPORTED | NOT_ESTABLISHED | 2/12 |
| pilot-02 | chain | 1 | NOT_ESTABLISHED | ERROR | 5/12 |

## pilot-01

- S01: Everyone who can pack a crate can seal a crate.
- S02: Bekele can paint a gate.
- S03: Everyone who can check a label can pack a crate.
- S04: Everyone who can paint a gate can clean a bench.
- S05: Hana can sort a parcel.
- S06: Genet can weave a mat.
- S07: Bekele can mentor Rahel.
- S08: Everyone who can sort a parcel can check a label.
- S09: Everyone who can weave a mat can fold a cloth.
- S10: Rahel can carry a drum.
- S11: Meron can make tea.
- S12: Bekele can supervise Dawit.

Learning: 351.478 seconds.

Parsed atoms (unaltered):

```json
[
  {
    "sentence_id": "S01",
    "text": "Everyone who can pack a crate can seal a crate.",
    "status": "success",
    "atoms": [],
    "error": null,
    "rejected_count": 1
  },
  {
    "sentence_id": "S02",
    "text": "Bekele can paint a gate.",
    "status": "success",
    "atoms": [
      "(: bekele_can_paint_gate (can_paint bekele gate) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S03",
    "text": "Everyone who can check a label can pack a crate.",
    "status": "success",
    "atoms": [
      "(: check_to_pack_rule (Implication (Premises (can_check_label $person)) (Conclusions (can_pack_crate $person))) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S04",
    "text": "Everyone who can paint a gate can clean a bench.",
    "status": "success",
    "atoms": [],
    "error": null,
    "rejected_count": 1
  },
  {
    "sentence_id": "S05",
    "text": "Hana can sort a parcel.",
    "status": "success",
    "atoms": [
      "(: hana_can_sort_parcel (can_sort hana parcel) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S06",
    "text": "Genet can weave a mat.",
    "status": "success",
    "atoms": [
      "(: genet_can_weave_mat (can_weave genet mat) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S07",
    "text": "Bekele can mentor Rahel.",
    "status": "success",
    "atoms": [
      "(: bekele_can_mentor_rahel (can_mentor bekele rahel) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S08",
    "text": "Everyone who can sort a parcel can check a label.",
    "status": "success",
    "atoms": [
      "(: sort_to_check_rule (Implication (Premises (can_sort $person parcel)) (Conclusions (can_check_label $person))) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S09",
    "text": "Everyone who can weave a mat can fold a cloth.",
    "status": "success",
    "atoms": [
      "(: weave_to_fold_rule (Implication (Premises (can_weave $person mat)) (Conclusions (can_fold $person cloth))) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S10",
    "text": "Rahel can carry a drum.",
    "status": "success",
    "atoms": [
      "(: rahel_carry_drum (can_carry rahel drum) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S11",
    "text": "Meron can make tea.",
    "status": "success",
    "atoms": [
      "(: meron_can_make_tea (can_make meron tea) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S12",
    "text": "Bekele can supervise Dawit.",
    "status": "success",
    "atoms": [
      "(: bekele_supervise_dawit (can_supervise bekele dawit) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  }
]
```

### pilot-01-supported

Can Hana sort a parcel?

Oracle: SUPPORTED; exact depth: 0.
Result: SUPPORTED; answer time: 47.518 seconds.

Used IDs: ['S05']

```json
{
  "question": "Can Hana sort a parcel?",
  "pln_query": "(: $prf (can_sort hana parcel) $tv)",
  "original_query": "(: $prf (can_sort hana parcel) $tv)",
  "executed_query": "(: $prf (can_sort hana parcel) $tv)",
  "fallback_used": false,
  "query_status": "well_aligned",
  "raw_proof": "['(: hana_can_sort_parcel (can_sort hana parcel) (STV 1.0 1.0))']",
  "sources": [],
  "answer": "Yes, Hana can sort a parcel.",
  "candidate_count": 1,
  "candidate_count_tried": 1,
  "executed_candidate_index": 0,
  "retry_used": false,
  "context_retrieval_seconds": 0.1124,
  "parse_query_seconds": 44.4135,
  "reasoning_seconds": 0.0004,
  "source_lookup_seconds": 0.0,
  "answer_generation_seconds": 2.9899
}
```

[Full worker output and logs](pilot-01/pilot-01-supported/output.json)

### pilot-01-control

Can Meron sort a parcel?

Oracle: NOT_ESTABLISHED; exact depth: null.
Result: NOT_ESTABLISHED; answer time: 65.553 seconds.

Used IDs: []

```json
{
  "question": "Can Meron sort a parcel?",
  "pln_query": "(: $prf (can_sort meron parcel) $tv)",
  "original_query": "(: $prf (can_sort meron parcel) $tv)",
  "executed_query": "(: $prf (can_sort meron parcel) $tv)",
  "fallback_used": false,
  "query_status": "well_aligned",
  "raw_proof": "[]",
  "sources": [],
  "answer": "I don't know \u2014 no proof was found for this question.",
  "candidate_count": 1,
  "candidate_count_tried": 1,
  "executed_candidate_index": 0,
  "retry_used": false,
  "context_retrieval_seconds": 0.1141,
  "parse_query_seconds": 64.8802,
  "reasoning_seconds": 0.5581,
  "source_lookup_seconds": 0.0,
  "answer_generation_seconds": 0.0
}
```

[Full worker output and logs](pilot-01/pilot-01-control/output.json)

## pilot-02

- S01: Everyone who can paint a gate can clean a bench.
- S02: Everyone who can sort a parcel can check a label.
- S03: Everyone who can weave a mat can fold a cloth.
- S04: Rahel can carry a drum.
- S05: Bekele can supervise Dawit.
- S06: Genet can weave a mat.
- S07: Abebe can sort a parcel.
- S08: Bekele can mentor Rahel.
- S09: Everyone who can check a label can pack a crate.
- S10: Bekele can paint a gate.
- S11: Everyone who can pack a crate can seal a crate.
- S12: Meron can make tea.

Learning: 273.891 seconds.

Parsed atoms (unaltered):

```json
[
  {
    "sentence_id": "S01",
    "text": "Everyone who can paint a gate can clean a bench.",
    "status": "success",
    "atoms": [],
    "error": null,
    "rejected_count": 1
  },
  {
    "sentence_id": "S02",
    "text": "Everyone who can sort a parcel can check a label.",
    "status": "success",
    "atoms": [],
    "error": null,
    "rejected_count": 1
  },
  {
    "sentence_id": "S03",
    "text": "Everyone who can weave a mat can fold a cloth.",
    "status": "success",
    "atoms": [],
    "error": null,
    "rejected_count": 1
  },
  {
    "sentence_id": "S04",
    "text": "Rahel can carry a drum.",
    "status": "success",
    "atoms": [
      "(: rahel_carry_drum (CanCarry rahel drum) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S05",
    "text": "Bekele can supervise Dawit.",
    "status": "success",
    "atoms": [
      "(: bekele_supervise_dawit (CanSupervise bekele dawit) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S06",
    "text": "Genet can weave a mat.",
    "status": "success",
    "atoms": [
      "(: genet_can_weave_mat (CanWeave genet mat) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S07",
    "text": "Abebe can sort a parcel.",
    "status": "success",
    "atoms": [
      "(: abebe_can_sort (CanSort abebe parcel) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S08",
    "text": "Bekele can mentor Rahel.",
    "status": "success",
    "atoms": [],
    "error": null,
    "rejected_count": 1
  },
  {
    "sentence_id": "S09",
    "text": "Everyone who can check a label can pack a crate.",
    "status": "success",
    "atoms": [],
    "error": null,
    "rejected_count": 1
  },
  {
    "sentence_id": "S10",
    "text": "Bekele can paint a gate.",
    "status": "success",
    "atoms": [
      "(: bekele_paint_gate (CanPaint bekele gate) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S11",
    "text": "Everyone who can pack a crate can seal a crate.",
    "status": "success",
    "atoms": [
      "(: pack_seal_rule (Implication (Premises (CanPack $agent crate)) (Conclusions (CanSeal $agent crate))) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  },
  {
    "sentence_id": "S12",
    "text": "Meron can make tea.",
    "status": "success",
    "atoms": [
      "(: meron_can_make_tea (CanMake meron tea) (STV 1.0 1.0))"
    ],
    "error": null,
    "rejected_count": 0
  }
]
```

### pilot-02-supported

Can Abebe check a label?

Oracle: SUPPORTED; exact depth: 1.
Result: NOT_ESTABLISHED; answer time: 76.906 seconds.

Used IDs: []

```json
{
  "question": "Can Abebe check a label?",
  "pln_query": "(: $prf (CanCheck abebe label) $tv)",
  "original_query": "(: $prf (CanCheck abebe label) $tv)",
  "executed_query": "(: $prf (CanCheck abebe label) $tv)",
  "fallback_used": false,
  "query_status": "well_aligned",
  "raw_proof": "[]",
  "sources": [],
  "answer": "I don't know \u2014 no proof was found for this question.",
  "candidate_count": 1,
  "candidate_count_tried": 1,
  "executed_candidate_index": 0,
  "retry_used": false,
  "context_retrieval_seconds": 0.116,
  "parse_query_seconds": 76.2544,
  "reasoning_seconds": 0.5346,
  "source_lookup_seconds": 0.0,
  "answer_generation_seconds": 0.0
}
```

[Full worker output and logs](pilot-02/pilot-02-supported/output.json)

### pilot-02-control

Can Meron check a label?

Oracle: NOT_ESTABLISHED; exact depth: null.
Result: ERROR; answer time: 0.591 seconds.

Used IDs: []

```json
{
  "question": "Can Meron check a label?",
  "pln_query": "",
  "original_query": "",
  "executed_query": "",
  "fallback_used": false,
  "query_status": "no_query",
  "raw_proof": "",
  "sources": [],
  "answer": "I couldn't translate this question into a logical query.",
  "candidate_count": null,
  "candidate_count_tried": null,
  "executed_candidate_index": null,
  "retry_used": null,
  "context_retrieval_seconds": 0.119,
  "parse_query_seconds": 0.4708,
  "reasoning_seconds": 0.0,
  "source_lookup_seconds": 0.0,
  "answer_generation_seconds": 0.0
}
```

[Full worker output and logs](pilot-02/pilot-02-control/output.json)

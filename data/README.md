# Bundled data

Everything the table scripts in `experiments/` read. About 13 MB in total.

| path | size | what it holds |
|---|---|---|
| `inventory.json` | 0.5 MB | the corpus as `{"child_parents": {child: [parents]}}`, 10,505 parent and child pairs over 8,218 children |
| `screened.json` | 1.1 MB | the screened cohort, 1,500 queries with their acquired candidate pools and known-direction labels |
| `uniform64.json` | 1.0 MB | the unscreened cohort with 64 uniformly sampled names per query, 512 queries |
| `lexical64.json` | 1.0 MB | the same 512 queries with 64 lexically similar names per query |
| `scores/<pool>_<teacher>.json.gz` | 3.1 MB | known-direction scores (eq. 1) of every candidate, one file per pool and teacher |
| `scores/<pool>_<teacher>_reverse.json.gz` | 5.6 MB | reverse scores, context-only name scores and token counts for the DC forms (eq. 3) |
| `predictions/<pool>_<labels>_seed<n>.json.gz` | 0.3 MB | stored answers of the trained students behind Table 1, one file per label source and seed |

The facts come from Wikidata and are released under CC0. Every name in the inventory and the pool
files is a public Wikidata entry. The predictions are student outputs for the same queries. Teachers are Qwen3-8B,
OLMo-2-7B, Mistral-7B-v0.3 and Llama-3.1-8B-Instruct, pinned in `configs/teachers.json`.

## Formats

A pool file.

```json
{"name": "screened",
 "queries": [{"id": "screened_0000", "parent": "...", "child": "...",
              "candidates": ["...", "..."], "label": "..."}],
 "withheld_children": ["..."],
 "evaluation_ids": ["screened_0001", "..."]}
```

`child` is the recorded answer and is used only for evaluation and for the surname strata. `label` is
the known-direction pseudo-label, which `ddv validate` rebuilds from the score files.
`withheld_children` lists the children whose forward facts are removed before training, and
`evaluation_ids` lists the exposure-free queries (1,390 of the screened cohort, all 512 of each
unscreened list).

A known-direction score file holds `teacher`, a `candidate_fingerprint` of the pool it scores, and
`scores`, one list per query in candidate order. A reverse file adds `reverse_mean` and
`reverse_tokens` per query, `parent_tokens` per query, and `context_mean` and `context_tokens` per
candidate name. `ddv.load_scores` checks the fingerprint and builds the six score forms of
Section 2.2 from these files.

A predictions file holds `{"pool", "labels", "seed", "predictions": [{"id", "text"}]}` for the
evaluated queries of one student run.

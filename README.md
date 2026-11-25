# AASL
ASL: AI-Augmented Supervised Learning

Input : text units U; theoretical domain D; seed categories S; model M; threshold τ
Output: final category schema C*; coded text units X

Each line is implemented at a marked place in `src/aasl/pipeline.py`;
`docs/algorithm\\\_mapping.md` maps every line to its
code and test and lists the operational choices the algorithm leaves open.
Install
```bash
pip install -e ".\\\[openai]"        # or .\\\[anthropic]; add .\\\[parquet] for parquet input
pip install -e ".\\\[dev]"           # tests
```
Python 3.9+. The core depends only on PyYAML.
Try it without an API key
```bash
aasl demo --run-dir runs/demo
```
The demo runs the whole algorithm on a synthetic corpus whose truth includes
two categories missing from the seeds, with a simulated model and simulated
coders. You will see round 1 fail calibration, recover after a prompt
revision, route the unseeded units to the residual, receive proposals (one
with fabricated quotes, which verification drops), keep one category and
reject the other, stop after round 2, pass held-out confirmation and code
all 3,000 units. Look in `runs/demo/` for the artefacts.

Use it on your data
Units. A CSV, JSONL or Parquet file with an id column, a text column
and metadata for stratification and clustering (e.g. firm, sector, region,
year).
Seeds. Write the domain definition, 3–5 seed categories with their
theory sources, and the attributes to extract, in a YAML file. See
`examples/cooperation/seeds.yaml`; its
seeds are illustrative only.
Config. Copy `examples/cooperation/config.yaml`,
point it at your files, pin a dated model snapshot, set τ and sample sizes
(`aasl sample-size` helps).
Run `aasl run your\\\_config.yaml`. The run stops whenever it needs
people (coding, resolving disagreements, revising prompts, adjudicating
proposals), tells you which files to complete, and resumes when you run
the command again. See `docs/workflow.md`.
Set `OPENAI\\\_API\\\_KEY` or `ANTHROPIC\\\_API\\\_KEY` in the environment.
Design
Decomposed coding. Three calls per unit: in-domain detection,
category assignment, attribute extraction. Each is evaluated on its own.
Subcategories and attributes. Categories can define subcategories (a second,
constrained call); attributes are extracted per category. Both are exported and
scored on units where human and model agree on the category.
An explicit residual. The model is told that `RESIDUAL` is a valid
answer and never forced into the nearest category. The residual is what
discovery works on and what schema coverage is measured from.
Per-category thresholds. Precision and recall must reach τ for every
category and for detection. Pooled scores are never used for the decision.
Categories with too few human-coded units cannot be certified; this is detected from
the human labels, before any model call, and the sample can be extended at random
without re-coding what is already coded.
Separated samples. The held-out sample is reserved before round 1 and
never used for calibration or discovery; each round calibrates on fresh
units. A ledger enforces this.
Verified proposals. A proposal's supporting text must appear verbatim
in the residual units it cites.
Researcher authority. Prompt revisions cannot change the category set;
only adjudication can, and every decision needs a written rationale.
Rejections are remembered. A rejected proposal is shown to the model with the
reason and dropped if it comes back.
Reliable yardstick. Inter-coder reliability (Krippendorff's alpha, Cohen's
kappa) is computed for every human-coded sample, with a minimum, and checked before
anyone resolves disagreements.
Auditability. Strict response schemas, retries, recorded coding errors (and a stop
when most calls fail), evidence quotes with verification flags, versioned schemas
with fingerprints, a model-reply cache keyed on every model setting, checkpoints,
an event log, an `audit.json` for the scaled coding, and a guard that refuses to
resume a run whose seed schema or corpus changed.


How to run AASL algorithm:

1. Install and set the key

bash
pip install -e ".[openai]"          # or ".[anthropic]"
export OPENAI_API_KEY=...           # or ANTHROPIC_API_KEY (Windows: set)

2. Prepare three things

Units file: CSV, JSONL or Parquet with an id column, a text column, and metadata such as firm, sector, region.
Seeds YAML: copy examples/cooperation/seeds.yaml and replace it with your domain definition, 3 or more theory-based categories, their sources, and attributes.
Config: copy examples/cooperation/config.yaml and edit the data path and columns, the schema path, model.name, the sample sizes, strata, and cluster (the firm column).

3. Pilot first. Use about 2,000 units and small sample sizes to test the API, then switch to the full file.

4. Run

bash
aasl run path/to/config.yaml

It stops whenever it needs people and prints which files to complete. Fill them in and run the same command again to resume. It repeats these stops in order:

Human coding (<run_dir>/human/coding/round1_calibration/). Two coders independently fill coder_1.csv and coder_2.csv using codebook.md, then save as CSV. A low alpha stops the run and writes RELIABILITY.txt; fix the codebook and recode.
Disagreements. Fill the final:* columns in resolution.csv.
Too few units of a category. If it reports a shortfall, raise calibration_size or holdout_size in the config and rerun. Units already coded are kept.
Below the threshold. Read human/revisions/<tag>/report.md and edit revision.yaml (clarify definitions, set ready: true).
Proposed categories. Edit human/adjudication/<tag>/proposals.yaml (keep, merge or reject, each with a rationale, then ready: true).
Next rounds. Rounds repeat steps 1-5 until no new category is kept.
Held-out check. The coders code human/coding/holdout/. If it passes, the model codes every unit.

5. Results are in <run_dir>/output/: coded_units.csv, final_schema.yaml, codebook.md, audit.json. The reports are in <run_dir>/reports/.



Package layout
```
src/aasl/
  core.py          units, labels, codings, reserved labels, exceptions
  schema.py        domain, categories, attributes, notes; versions and fingerprints
  prompts.py       detect / assign / attributes / propose prompts and response schemas
  coder.py         the model coder M (lines 5, 14)
  metrics.py       per-category evaluation, coverage, bootstrap CIs, sample size
  calibration.py   inner loop (lines 4–7) and prompt revisers
  discovery.py     residual and proposals (lines 8–9)
  adjudication.py  decisions and schema update (lines 10–11)
  annotation.py    human coding files and resolution (lines 3, 13)
  reliability.py   kappa, alpha, agreement
  sampling.py      stratified, cluster-capped, extendable samples and the sample ledger
  pipeline.py      Algorithm 1
  llm/             model interface, cache and retries, OpenAI and Anthropic adapters
  simulate.py      synthetic corpus, simulated model and coders (tests, demo)
tools/
  mutation\\\_check.py   breaks the implementation 33 ways; the tests must catch each
  cli.py           aasl run | status | demo | sample-size | codebook
```
Tests and verification
```bash
pytest                              # 86 tests, \\\~6 s
python tools/mutation\\\_check.py      # 33 deliberate breakages; every one must be caught
```
The suite covers each module and runs the full algorithm end to end, both with
simulated coders and through the file-based human workflow (pause, complete files,
resume). Beyond the tests, the implementation was checked as follows:
Mutation check. `tools/mutation\\\_check.py` weakens one property at a time (a
held-out leak, no residual option, unverified proposals, a disabled threshold or
reliability gate, extension reusing units, and so on). All 33 are caught.
Reliability statistics match `scikit-learn` (kappa) and the `krippendorff` package
(alpha) to machine precision on random data, and alpha reproduces Krippendorff's
published reference example (0.743).
Response validation agrees with the `jsonschema` library on 12,000 randomly
corrupted replies, and every response schema satisfies OpenAI's strict-mode rules.
Python versions. The suite passes on Python 3.9 and 3.12.
Stability. The full algorithm reaches the expected outcome (unseeded category found
and kept, decoy rejected, held-out passed, all units coded) on 25 of 25 synthetic
corpora.


"""Mutation check: break the implementation in specific ways and confirm the test suite notices.

Each mutant removes or weakens one property that Algorithm 1 (or a later safeguard)
depends on: a held-out leak, a forced choice with no residual, unverified proposals, a
disabled threshold gate, and so on. A suite that stays green on a mutant would not
protect that property. Run from the repository root:

    python tools/mutation_check.py

This is a development aid. Mutants are exact-text replacements, so after refactoring the
source a pattern may stop matching; the script then says which one to update.
"""
import os, pathlib, shutil, subprocess, sys, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"; TESTS = str(ROOT / "tests")
M = [
 ("M01 held-out can leak into calibration", "sampling.py", "exclude = self.used() if fresh else self.holdout_uids()", "exclude = set()"),
 ("M02 no RESIDUAL option (forced choice)", "prompts.py", "enum = schema.category_names + [RESIDUAL]", "enum = schema.category_names"),
 ("M03 only detection gated, categories ignored", "metrics.py", "gated = [detection] + list(categories.values())", "gated = [detection]"),
 ("M04 proposal quotes not verified", "discovery.py", "ok = uid in allowed and quote_in_text(s[\"quote\"], units[uid].text)", "ok = uid in allowed"),
 ("M05 prompt revision may change categories", "calibration.py", "if revised.category_names != schema.category_names:", "if False:"),
 ("M06 stop after round 1 regardless", "pipeline.py", "            if not kept:                                                      # line 12\n                break", "            break"),
 ("M07 never stop (ignore 'no new category kept')", "pipeline.py", "if not kept:                                                      # line 12", "if False:"),
 ("M08 held-out confirmation never blocks", "pipeline.py", "        if not report.passes:\n            self.store.log(\"confirmation_failed\"", "        if False:\n            self.store.log(\"confirmation_failed\""),
 ("M09 residual includes out-of-domain units", "discovery.py", "residual = sorted(c.uid for c in in_domain if c.label.category == RESIDUAL)", "residual = sorted(c.uid for c in ok if c.label.category in (RESIDUAL, None))"),
 ("M10 calibration accepts first pass regardless", "calibration.py", "if report.passes:                                                     # line 7", "if True:"),
 ("M11 proposals drawn from all units, not residual", "discovery.py", "    residual, n_dom, cov = collect_residual(codings)\n    proposals: dict", "    residual, n_dom, cov = collect_residual(codings)\n    residual = sorted(c.uid for c in codings)\n    proposals: dict"),
 ("M12 decisions need no rationale", "adjudication.py", "if not (d.rationale or \"\").strip():", "if False:"),
 ("M13 final coding uses seed schema, not C*", "pipeline.py", "codings = self._code(final, self.units, \"final\")", "codings = self._code(self.seed_schema, self.units, \"final\")"),
 ("M14 human reliability gate disabled", "pipeline.py", "elif value < self.s.min_reliability:", "elif False:"),
 ("M15 min-support rule disabled", "metrics.py", "if s.support < min_support:\n            failures.append", "if False:\n            failures.append"),
 ("M16 checkpoints reused across schema versions", "coder.py", "if c.schema_fingerprint == fingerprint and c.status == \"ok\":", "if c.status == \"ok\":"),
 ("M17 coding errors silently mapped to out-of-domain", "coder.py", "return Coding(uid=unit.uid, label=None, status=\"error\"", "return Coding(uid=unit.uid, label=Label(in_domain=False), status=\"ok\""),
 ("M18 merge into existing category counts as a new category", "adjudication.py", "    return schema, kept_names, log", "    kept_names += [d.into for d in decisions if d.action == 'merge' and d.into in schema.category_names]\n    return schema, kept_names, log"),
 ("M19 FileAnnotator reliability gate disabled", "annotation.py", "if self.min_reliability is None or value is None or value >= self.min_reliability:\n            return", "if True:\n            return"),
 ("M20 support check disabled", "pipeline.py", "short = support_shortfalls(gold, schema, self.s.min_support)", "short = []"),
 ("M21 holdout support check removed", "pipeline.py", "self._require_support(h_ann.gold, final, \"holdout_size\", \"holdout\")", "pass"),
 ("M22 extension may reuse already-used units", "sampling.py", "self.draw(part, units, n - have, fresh=True, cluster_counts=counts, **kwargs)", "self.draw(part, units, n - have, fresh=False, cluster_counts=counts, **kwargs)"),
 ("M23 rejected proposals not remembered", "pipeline.py", "round_no=r, rejected=self._rejected())", "round_no=r, rejected=None)"),
 ("M24 seed/corpus guard disabled", "pipeline.py", "if self.store.has(\"run_inputs\"):", "if False:"),
 ("M25 Excel BOM not handled", "annotation.py", "text = Path(path).read_text(encoding=\"utf-8-sig\")", "text = Path(path).read_text(encoding=\"utf-8\")"),
 ("M26 empty citation flagged as unverified", "coder.py", "        if quote:\n            verified[key]", "        if True:\n            verified[key]"),
 ("M27 cache key ignores temperature/seed", "llm/openai_model.py", "self.name = f\"openai:{model}|t={temperature}|seed={seed}\"", "self.name = f\"openai:{model}\""),
 ("M28 fail-fast removed", "coder.py", "and len(errors) > FAIL_FAST_RATIO * len(batch)):", "and False):"),
 ("M29 subcategory step skipped", "coder.py", "if self.schema.category(category).subcategories:\n                        sub", "if False:\n                        sub"),
 ("M30 subcategory scoring ignores category disagreement", "metrics.py", "\n                    or c.label.category != cat.name):", "):"),
 ("M31 programmatic rejected-drop disabled", "discovery.py", "if k in turned_down:", "if False:"),
 ("M32 stale-resolution carry-over disabled", "annotation.py", "carried = {r[\"uid\"]: r for r in old if r.get(\"uid\") in disputed}", "carried = {}"),
 ("M33 reliability gate runs after resolution", "annotation.py", "        self._gate_reliability(folder, tag, reliability, units, flat, disagree)\n        gold = {u: per_coder[0][u] for u in uids}\n        if disagree:\n            gold.update(self._resolve(folder, units, per_coder, disagree, schema, tag))", "        gold = {u: per_coder[0][u] for u in uids}\n        if disagree:\n            gold.update(self._resolve(folder, units, per_coder, disagree, schema, tag))\n        self._gate_reliability(folder, tag, reliability, units, flat, disagree)"),
]
survived = []
for name, f, old, new in M:
    d = pathlib.Path(tempfile.mkdtemp(prefix="aasl_mut_")); shutil.rmtree(d); shutil.copytree(SRC, d)
    p = d/"aasl"/f; s = p.read_text()
    assert s.count(old) == 1, f"{name}: source pattern matched {s.count(old)}x (update tools/mutation_check.py)"
    p.write_text(s.replace(old, new))
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", TESTS, "-p", "no:cacheprovider"],
                       env={**os.environ, "PYTHONPATH": str(d)}, capture_output=True, text=True, timeout=600)
    last = [l for l in r.stdout.splitlines() if l.startswith("FAILED") or "passed" in l or "failed" in l]
    status = "KILLED " if r.returncode != 0 else "SURVIVED"
    if r.returncode == 0: survived.append(name)
    print(f"{status} {name:62s} {last[0][:90] if last else ''}")
print(f"\n{len(M)-len(survived)}/{len(M)} mutants killed; survivors: {survived or 'none'}")

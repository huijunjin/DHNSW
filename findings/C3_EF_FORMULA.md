# C3 follow-up — fixing Eq. (1) instead of conceding it

C3 recorded that `ef_ref = ef_base + (d / alpha)^2` produces 1000 at 3072
dimensions and that DHNSW then inherits the cost of an operating point nobody
chose. Reporting that as a limitation is honest but weak. This asks whether the
formula is repairable, and it is.

```bash
python scripts/run_c3_ef_formula.py --phase 3 \
    --datasets sift1m glove100k landmark-dino landmark-nomic mnist gist simplewiki-openai \
    --formulas paper hardcap200 softcap200 log intrinsic intrinsic_scaled none
# -> results/phase3/c3_ef_formula.csv
```

Seven candidates, seven datasets from 128d to 3072d, 60,000 vectors each, run
against the C++ port. A candidate has to clear two bars, not one:

  (i)  it must not blow up at 3072d, and
  (ii) it must leave 128-960d where the paper already validated it.

Clearing only (i) trades a validated range for an unvalidated one.

## Criterion (ii) — ef_ref in the validated range

| dataset | dim | paper | hardcap200 | softcap200 | log | intrinsic_scaled | intrinsic / none |
|---|---:|---:|---:|---:|---:|---:|---:|
| sift1m | 128 | 101 | **101** | 102 | 128 | 114 | 100 |
| glove100k | 300 | 109 | **109** | 109 | 148 | 206 | 100 |
| landmark-dino | 768 | 149 | **149** | 151 | 174 | 396 | 100 |
| landmark-nomic | 768 | 149 | **149** | 151 | 174 | 209 | 100 |
| mnist | 784 | 149 | **149** | 153 | 175 | 201 | 100 |
| gist | 960 | 181 | **181** | 174 | 181 | 181 | 100 |
| **max deviation** | | 0 | **0** | 7 | 39 | 247 | 81 |

`hardcap200` is identical to Eq. (1) here by construction: `(d // 100)^2` stays
under the ceiling for every `d` below 1414, so the cap is inert in exactly the
range the paper measured.

## Criterion (i) — SimpleWiki-OpenAI at 3072d

| formula | ef_ref | build vs vanilla | recall vs vanilla | recall gained per unit of build cost |
|---|---:|---:|---:|---:|
| paper | 1000 | +482.7 % | +0.88 %p | 1.82 |
| **hardcap200** | **300** | **+151.8 %** | **+0.70 %p** | **4.61** |
| softcap200 | 298 | +150.4 % | +0.72 %p | 4.79 |
| log | 219 | +99.0 % | +0.71 %p | 7.17 |
| intrinsic_scaled | 282 | +140.8 % | +0.67 %p | 4.76 |
| intrinsic | 100 | +6.4 % | +0.01 %p | 1.57 |
| none | 100 | +6.2 % | +0.14 %p | 22.73 |

**A ceiling of 200 buys 80 % of the paper's recall gain for 31 % of its build
cost.** Only `hardcap200` clears both bars, so that is the recommendation: one
`min()` in Eq. (1), no change anywhere the paper was validated.

## What the intrinsic dimension actually says

The two-NN estimator was the interesting hypothesis and it failed, informatively.

| dataset | ambient | two-NN | PCA-95 |
|---|---:|---:|---:|
| sift1m | 128 | 5.2 | — |
| glove100k | 300 | 14.5 | — |
| landmark-nomic | 768 | 14.7 | — |
| mnist | 784 | 14.2 | 148 |
| gist | 960 | 12.7 | 292 |
| landmark-dino | 768 | 24.3 | — |
| simplewiki-openai | 3072 | **19.1** | 513 |

**3072 ambient dimensions carry an intrinsic dimension of 19.** That is the
cleanest evidence that ambient `d` was the wrong variable: SimpleWiki is the
widest dataset here and nowhere near the hardest.

But intrinsic dimension does not *order* these datasets the way ambient
dimension does -- GIST at 960d estimates lower (12.7) than landmark-dino at
768d (24.3). So substituting it is not a recalibration of Eq. (1), it is a
different formula: refitting alpha on GIST sends landmark-dino from 149 to 396.
With alpha unchanged the term vanishes entirely (`(19 // 100)^2 = 0`), which is
why `intrinsic` and `none` produce identical numbers throughout.

## A correction to an earlier reading

C11 showed the C++ port's regression is entirely Eq. (1)'s ef increase, and the
obvious conclusion was to drop the term -- `dhnsw_ef100` builds faster than
vanilla on all four raw datasets. This sweep says that conclusion was too
strong. Dropping the term costs recall where the term was doing work: on GIST
`none` lands at **-0.88 %p against vanilla** while `paper` gains +0.96 %p, and
at 3072d it recovers only +0.14 %p of the +0.88 %p available.

**Eq. (1)'s direction was right and only its magnitude was wrong.** A cap keeps
the direction and removes the failure; removal throws away both.

## Caveats

One seed, one run per cell, 60,000 vectors per dataset (the paper's scale for
MNIST but a reduction for SIFT1M and GIST1M). The build-time percentages carry
the usual run-to-run noise established in C2, so the ranking between
`hardcap200`, `softcap200` and `intrinsic_scaled` at 3072d -- 150.4 to 151.8 %
apart -- should not be read as separating them. The gap that matters, 482.7 %
against ~150 %, is far outside that noise. The ceiling itself (200) is set by
the same kind of judgement alpha was, and this sweep does not tune it.

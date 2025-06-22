### Picking the “representative” path for checkpoints

Below is a practical recipe that yields (1) *clear, realistic checkpoints* and (2) a *single bold trajectory* that’s statistically most representative of all Monte-Carlo runs.

| Stage                               | What we do                                                                                                                                                                                                                  | Why it works                                                                                                                                |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **1.  Simulate**                    | Run *N* (\~2 000) Monte-Carlo trajectories using the bulk/cut rules and P-ratio distributions.                                                                                                                              | Gives the full cloud of plausible futures.                                                                                                  |
| **2.  Capture phase-change events** | For each run, record the timestamp & metrics at every phase transition:<br>• end-of-cut (min BF hit)<br>• end-of-bulk (max BF hit)<br>• …repeat until goal.                                                                 | These events are the *checkpoints* users care about.                                                                                        |
| **3.  Build “median checkpoints”**  | For each step *k* (Cut 1, Bulk 1, Cut 2, …):<br>• Take the **median** of time-to-event across all runs.<br>• Take the **median** of BF%, weight, lean, fat at that time.                                                    | This yields the most likely calendar dates & body-comp values for each step.                                                                |
| **4.  Assemble a median schedule**  | Stitch those median checkpoints together into a single phased timeline.                                                                                                                                                     | Now we have one clean sequence of dates + targets to display (glyph triangles ▲▼ in the plot and a step table).                             |
| **5.  Pick a “real” path to bold**  | Compute each run’s **total squared distance** from the median schedule (evaluate weight & BF% at shared weekly grid).<br>Choose the run with the *smallest* distance and draw it bold.                                      | ✔ Guarantees the bold line is **an actual MC trajectory** (not an impossible average) but is also the *closest* to the median expectations. |
| **6.  Visualise fan + bold path**   | • Shaded 10-90 % fan for context.<br>• Light random spaghetti lines (optional).<br>• **Bold line** = representative path.<br>• ▲ (top of bulks) and ▼ (end of cuts) placed at the *median checkpoints* with hover tooltips. | Users see the full uncertainty *and* a concrete “most-likely journey” they can mentally follow.                                             |

#### Why this is robust

1. **Phase-aware** – Median is taken *per phase*, so timing differences in long vs. short cuts don’t blur together.
2. **Real path bolded** – You avoid showing an impossible average; every point on the bold path is feasible under the model.
3. **Variance reflects training level** – Because σ shrinks for experienced lifters (and from the user’s scan-derived variance), the fan narrows automatically, making their checkpoints tighter.

#### Implementation sketch

```python
# after running simulations -> list[DataFrame] runs
phases = ["cut1", "bulk1", "cut2", ...]
median_cp = {}

for phase in phases:
    times  = [run.loc[run.phase == phase, 'week'].iloc[-1] for run in runs]
    bf     = [run.loc[run.phase == phase, 'bf'].iloc[-1]    for run in runs]
    weight = [run.loc[run.phase == phase, 'weight'].iloc[-1] for run in runs]
    median_cp[phase] = dict(
        week   = np.median(times),
        bf     = np.median(bf),
        weight = np.median(weight)
    )

# pick representative path
target_vec = interpolate_to_week_grid(median_cp)
best_idx   = np.argmin([
    np.sum((run_grid - target_vec)**2) for run_grid in all_run_grids
])
bold_run   = runs[best_idx]
```

---

**Result to user**

* **Step table**:

  | Phase      | ETA  | Weight | BF % | Lean Mass | Fat Mass |
  | ---------- | ---- | ------ | ---- | --------- | -------- |
  | End Cut 1  | 12 w | 175 lb | 18 % | …         | …        |
  | End Bulk 1 | 32 w | 188 lb | 25 % | …         | …        |
  | …          |      |        |      |           |          |

* **Plot**: Shaded fan, thin grey paths, bold navy path, ▲/▼ glyphs at the ETAs.

That gives a single, intuitive roadmap while still embracing the probabilistic philosophy championed by Stronger By Science.


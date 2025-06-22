# Product Requirements Document (PRD): Multi‑Phase Goals + Probabilistic Forecasting for RecompTracker

> **Purpose**  Replace today’s single‑step goal logic with an evidence‑based, multi‑phase bulk/cut roadmap **and** a Monte‑Carlo “fan‑chart” forecast. Users will see realistic checkpoints plus uncertainty bands, with two UX flows: **Quick Forecast** and **Advanced Planner**.

---

## 1 Problem Statement

*Current state*: RecompTracker projects a straight‑line jump from the latest scan to a target ALMI/FFMI percentile. Real progress happens via alternating bulks & cuts, and outcome variance is high—especially in novices.

*Need*: 

1. **Multi‑phase planning** (healthy‑BF cut ▶ bulk ▶ cut ▶ …) 
2. **Probabilistic ranges** rather than deterministic numbers 
3. Dual UX paths so beginners aren’t overwhelmed, while power users can customise every assumption.

---

## 2 High‑Level Goals

1. Generate an **automatic phased plan** (min‑BF → bulk peak → cut trough …) until percentile goal reached.
2. Run **Monte‑Carlo simulations** to visualise a cloud of possible futures.
3. Present a **bold representative path** + checkpoint table so users know “what’s most likely”.
4. Offer **Quick** vs **Advanced** flows.

---

## 3 Multi‑Phase Strategy Templates

RecompTracker offers **two starting templates**. The app chooses *Cut‑First* automatically when the user is above the healthy BF range, but in **Advanced Planner** mode the user can switch to a *Bulk‑First* template if they prefer to build muscle first (mirroring Stronger By Science’s flexible philosophy).

### 3.1 Cut‑First (Default) Template

| Step     | Rule                                         | Men (BF %)                 | Women (BF %)               | Duration Guardrail |
| -------- | -------------------------------------------- | -------------------------- | -------------------------- | ------------------ |
| 1 Cut    | Reach healthy range                          | <20 %                      | <30 %                      | min 8 wk           |
| 2 Bulk   | Stop at max acceptable BF                    | 15–18 %                    | 25–30 %                    | min 12 wk          |
| 3 Cut 2  | Return to desired BF                         | user choice (e.g. 12–15 %) | user choice (e.g. 22–25 %) | min 8 wk           |
| 4 Repeat | Loop bulk/cut until ALMI/FFMI percentile hit | —                          | —                          | —                  |

### 3.2 Bulk‑First (Optional) Template

| Step     | Rule                                      | Men (BF %)                            | Women (BF %)                          | Duration Guardrail |
| -------- | ----------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------ |
| 1 Bulk   | Start in small surplus, stop at bulk peak | +3–5 % BF from current or 18–22 % cap | +3–5 % BF from current or 28–32 % cap | min 12 wk          |
| 2 Cut 1  | Drop to preferred BF range                | user choice (e.g. 12–15 %)            | user choice (e.g. 22–25 %)            | min 8 wk           |
| 3 Bulk 2 | Resume surplus, respecting BF cap         | same as Step 1                        | same as Step 1                        | min 12 wk          |
| 4 Repeat | Alternate until ALMI/FFMI percentile hit  | —                                     | —                                     | —                  |

**Rate defaults** remain *MacroFactor* “happy‑medium” for bulks (\~0.3 % BW/wk) and *conservative* for cuts (\~0.5–0.75 lb/wk). The user can override rates and BF caps in **Advanced Planner** mode.

## 4 Probabilistic Forecast Engine Forecast Engine

### 4.1 Inputs

* Latest scan metrics
* Phase rules above
* Training status ➜ variability factor (σ): 0.50 novice · 0.25 intermediate · 0.10 advanced
* Empirical σ from recent scans (blended 50 %)

### 4.2 Monte‑Carlo Loop (per run)

```
while not goal:
    μ_fat, μ_lean = rate_lookup(state, phase)
    σ_fat  = k * μ_fat
    σ_lean = k * μ_lean
    dfat  = N(μ_fat, σ_fat)
    dlean = N(μ_lean, σ_lean)
    state.apply(dfat, dlean)
    if phase_complete(state): switch_phase()
```

\*2 000 iterations cached via \**`@st.cache_data`*

### 4.3 Checkpoints & Representative Path

1. For each phase, collect **end‑week** across runs and compute medians (time, BF %, weight).
2. Pick the simulated trajectory with **lowest RMS distance** to those medians → **bold path**.

### 4.4 Fan Chart Rendering

* Shaded 10–90 % & 25–75 % bands.
* Optional grey spaghetti (30 random runs).
* Bold representative line.
* ▲ at bulk peaks, ▼ at cut troughs (median schedule) with hover tooltips (week, weight, BF %, lean, fat).

---

## 5 Dual UX Flows

| Flow                 | Entry                      | Audience            | Key Traits                                                                                                                                   |
| -------------------- | -------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Quick Forecast**   | “Run Analysis”             | New users           | Fixed 2 000 runs, 80 % fan, minimal controls                                                                                                 |
| **Advanced Planner** | “Advanced” button / wizard | Coaches & tinkerers | Multi‑step wizard: current → prefs → training override → sim settings → preview; controls for run count, P‑ratio overrides, spaghetti toggle |

Both flows share the same fan chart & checkpoints; Advanced exposes more knobs.

---

## 6 Scientific Support

* **Novice recomp ranges**: 4–7 lb lean gain + 12–18 lb fat loss in 12 wk (Longland 2016; Hartman 2007, etc.).
* \*\***P‑ratio reference ranges (lean mass share of weight change)**
  • **Bulk (any BF%)**: **0.45 – 0.50** lean, 0.50 – 0.55 fat
  • **Cut, high BF** (> 30 %F / > 25 %M): **0.20 – 0.25** lean, 0.75 – 0.80 fat
  • **Cut, moderate BF** (20 – 30 %): **0.30 – 0.40** lean, 0.60 – 0.70 fat
  *Sources: Hall 2007, Forbes 1987 partitioning models; Stronger By Science P‑ratio review (Trexler & Nuckols 2021).*
* Variance shrinks with training age; personalised via user scan history.

---

## 7 Implementation Tasks

1. **Simulation & Test‑Driven Core (no UI)**

   * Build `mc_forecast.py` (vectorised NumPy Monte‑Carlo engine).
   * Create *canned profiles* test‑suite → novice + overweight, novice + lean, intermediate, advanced.
   * Assert that both **Cut‑First** and **Bulk‑First** templates generate sensible *median checkpoint* schedules.
   * CI: tests must pass before any UI work begins.
2. **Expose Forecast API**

   * Add `forecast_core.get_plan(user_state)` returning fan percentiles, representative path, and median checkpoints.
3. **UI Upgrade Phase**

   * Wire Quick Forecast panel to new API; show fan chart & step table.
   * Build Advanced Planner wizard (run‑count slider, P‑ratio override, etc.).
   * Add glyphs ▲ ▼ using median checkpoints.
4. **Docs & Tooltips**

   * Inline explanations (probabilistic mindset, representative path).
   * Add modal explainer copy (bulking vs cutting) referencing Stronger By Science P‑ratio article + MacroFactor bulking/cutting series; update README & in‑app help.

##


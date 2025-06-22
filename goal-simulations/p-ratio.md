There’s no universally accepted single “P‑ratio,” but several well-cited scientific models help us understand how changes in fat and lean mass relate and how the partitioning effect varies by individual and context:

---

## 📊 P‑Ratio Foundations: Engineering the Partition

### 1. **Forbes & Hall Model (2007)**

* Kevin Hall (2007) built on Forbes’ work, showing that the **P‑ratio (lean mass gained or lost per lb/kg of weight change)** depends strongly on **initial body composition**([pubmed.ncbi.nlm.nih.gov][1]).
* With **higher initial fat**, more weight change is fat; **lower initial fat** yields a higher share of lean in weight changes—all in weight gain *and* loss contexts.

### 2. **Satiety and Set-Points Model (Dulloo & Jacquet)**

* Explains why, after dieting, the body **preferentially restores fat**, sometimes overshooting previous levels due to adaptive fat storage mechanisms.
* While not prescribing a P‑ratio, it shows that partitioning is influenced by **physiological memory** and fat levels.

---

## 🧬 Estimated Ranges in Studies & Practice

* **Forbes/Hall Equation**: P‑ratio ranges from \~0.2–0.25 in **obese subjects** to \~0.5 in **leaner individuals**—this reflects lean mass *lost* during weight loss([pubmed.ncbi.nlm.nih.gov][1]).
* In **overfeeding studies**, Forbes found roughly **44–51%** of weight gain went to lean mass (P‑ratio ≈ 0.44–0.51)([pubmed.ncbi.nlm.nih.gov][1]).
* **Stronger by Science analysis** (Eric Trexler & Greg Nuckols) found **no strong evidence** that P‑ratio worsens at higher BF%. They suggest any body-fat impact is minor and poorly supported([strongerbyscience.com][2]).

---

## 🎯 Relevance for RecompTracker

### 1. **Initial Cut (Weight Loss)**

* Model P‑ratio based on **starting body fat**:

  * **High BF (30%+ women, 25%+ men)** → expect P ≈ 0.2–0.25 (20–25% of weight loss is lean mass)
  * **Moderate BF (20–30%)** → P ≈ 0.30–0.40

### 2. **Bulk (Weight Gain)**

* Expect roughly **45–50% of weight gain to be lean mass** (P ≈ 0.45–0.50), per Forbes overfeeding data.

### 3. **Evidence Limitations**

* P‑ratio is individual-dependent—it’s not a static property across everyone.
* Strong evidence supports using **value ranges**, not absolutes.
* The hypothesis that leaner individuals always have better P‑ratios is **not well supported**([cambridge.org][3], [mennohenselmans.com][4]).

---

## ✅ Recommendations for RecompTracker

* Implement a **P‑ratio lookup** based on user’s BF% when forecasting:

  * *If in deficit (cut)*: use 0.2–0.4 depending on initial BF%.
  * *If in surplus (bulk)*: use \~0.45–0.50 lean gain ratio.
* Display ranges and note individual variability.
* Allow users to toggle to *lean*, *moderate*, or *balanced* partitioning to reflect personal differences or research uncertainty.
* Include a tooltip: “P‑ratio estimates based on Forbes/Hall partitioning data; actual results may vary.”

---

### 🧾 Summary Table

| Phase   | Initial BF%       | P‑Ratio Estimate | Implications                                               |
| ------- | ----------------- | ---------------- | ---------------------------------------------------------- |
| Deficit | High (>30/25%)    | \~0.20–0.25      | Expect \~75–80% of weight loss as fat, 20–25% lost as lean |
| Deficit | Moderate (20–30%) | \~0.30–0.40      | More favorable lean preservation                           |
| Surplus | N/A               | \~0.45–0.50      | \~45–50% of weight gain as muscle, rest as fat             |

---

Would you like me to integrate this P‑ratio logic into the PRD as conditional defaults or draft the actual implementation snippets for the planning module?

[1]: https://pubmed.ncbi.nlm.nih.gov/17367567/?utm_source=chatgpt.com "Body fat and fat-free mass inter-relationships: Forbes's theory revisited"
[2]: https://www.strongerbyscience.com/p-ratios-rebuttal/?utm_source=chatgpt.com "Body-Fat and P-Ratios: A Rebuttal To The Rebuttal"
[3]: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/6203E3D9BE27B170E2C70025BC47E278/S0007114598000166a.pdf/partitioning_between_protein_and_fat_during_starvation_and_refeeding_is_the_assumption_of_intraindividual_constancy_of_pratio_valid.pdf?utm_source=chatgpt.com "Partitioning between protein and fat during starvation and refeeding"
[4]: https://mennohenselmans.com/optimal-body-fat-muscle-growth/?utm_source=chatgpt.com "What's the optimal body fat range for muscle growth?"


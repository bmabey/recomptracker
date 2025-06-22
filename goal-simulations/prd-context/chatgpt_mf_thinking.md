# Summary of MacroFactor Articles and Recommendations for RecompTracker

## 📚 Article Summaries

### 1. **Should I Bulk or Cut?**
- **Default advice**: Choose the path that excites you more.
- If unsure, **bulking** or **maintenance** is typically better for long-term muscle gain.
- **Cutting is recommended** if:
  - Men: >25% body fat
  - Women: >35% body fat
  - You have aesthetic or sport-specific goals
- **Bulking is recommended** if:
  - You want to build more muscle
  - You're below the above thresholds
- **Maintenance** allows for body recomposition (fat loss + muscle gain) at a slower rate.

---

### 2. **How Fast to Gain Weight When Bulking**
- **Rate depends on training experience**:

| Training Level | Conservative | Medium     | Aggressive | Max (lbs/week) |
|----------------|--------------|------------|------------|----------------|
| Beginner       | 0.2%         | 0.5%       | 0.8–1.0%   | 0.9–1.8        |
| Intermediate   | 0.15%        | 0.325%     | 0.575%     | 0.6–1.4        |
| Experienced    | 0.1%         | 0.15%      | 0.35–0.6%  | 0.2–1.0        |

- Gaining faster **does not proportionally improve muscle gain** in experienced lifters.
- **Best lean mass retention** occurs at ≤0.3–0.4% body weight/week.

---

### 3. **How Fast to Lose Weight When Cutting**
- **Recommended range**: 0.25%–1% of body weight/week  
  (≈ 0.5–1.5 lbs/week for most people)
- Below ~0.8%/week = better muscle retention
- **Upper bound**: 2 lbs/week; beyond this, muscle loss and adherence issues increase
- Adjust rate based on:
  - Mood
  - Hunger
  - Performance
  - Motivation

---

## 💡 Feature Suggestions for RecompTracker App

### 1. **Bulk/Cut Decision Support**
- Analyze BF% from scans
- Auto-suggest: “Consider bulking/cutting/maintaining based on current body fat levels and goals”

---

### 2. **Rate of Change Recommendations**
- Add bulking/cutting rate suggestions in goal panel:
  - User selects: Training level + aggressiveness
  - Output: Weekly target weight change + estimated lean/fat breakdown

> Example: “Gain 0.33 lbs/week → ~85% lean mass gain (Beginner, Conservative)”

---

### 3. **Forecasting Timelines**
- When user sets a goal percentile:
  - Estimate time to reach it at different gain/loss rates
  - Visual overlays on percentile plots (dotted trajectory)

---

### 4. **Recomposition Detection**
- Detect if user is gaining lean mass while losing fat
- Flag as “body recomposition”
- Recommend staying at maintenance temporarily

---

### 5. **Motivation-Based Goal Adjustment**
- Let users optionally log:
  - Mood, adherence, energy
- If motivation drops, suggest reducing aggressiveness of bulk/cut

---

Would you like mock UI layouts, Streamlit code snippets, or implementation plans for any of these?


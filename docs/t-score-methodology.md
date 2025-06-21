# T-Score Methodology Documentation

## Overview

T-scores are a standardized measurement that compares an individual's muscle mass to a "peak reference population" - typically healthy young adults aged 20-30. T-scores tell you how many standard deviations you are from the peak values typically seen in young adulthood.

## T-Scores in Medical Context

### Bone Density vs. Muscle Mass

T-scores are **commonly used** for bone density analysis, where they help diagnose osteoporosis and fracture risk. The World Health Organization officially defines osteoporosis as a bone density T-score of -2.5 or lower.

For muscle mass and ALMI, T-scores are **much less standard**. While some research has explored T-score approaches for sarcopenia (muscle loss) assessment, there are no official clinical guidelines or widely accepted thresholds like there are for bone density.

### Clinical Interpretation

In simple terms: **T-scores show how your current muscle mass compares to what you might have had in your physical prime.**

- **T-score of 0**: Your muscle mass matches the average healthy 25-year-old
- **T-score of +2**: You have exceptional muscle mass - better than 97% of young adults at their peak
- **T-score of -2**: Your muscle mass is significantly below typical young adult levels

## RecompTracker T-Score Zones

We've created 5 experimental zones that are **not based on clinical standards**:

- **Elite Zone** (T ≥ +2.0): Exceptional muscle mass
- **Peak Zone** (0 ≤ T < +2.0): Excellent muscle mass
- **Approaching Peak** (-1.0 ≤ T < 0): Good muscle mass
- **Below Peak** (-2.0 ≤ T < -1.0): Below optimal
- **Well Below Peak** (T < -2.0): Significantly below optimal

## Important Disclaimer

**Age-appropriate percentiles remain the recommended approach** for actual goal-setting and health assessment. T-scores are provided as an experimental feature for those interested in comparing against peak young adult muscle mass.

Think of T-scores as a "fun fact" overlay rather than clinical guidance - your primary focus should remain on improving within your age and gender demographic using the standard percentile system.

## Technical Implementation

### The Challenge: From LMS Parameters to Population Statistics

When implementing T-score functionality for body composition analysis, we encountered a fundamental problem that highlights the difference between **statistical parameters** and **actual population statistics**. This section explains how we solved it using Monte Carlo sampling from LMS distributions.

### The Initial Problem: Incorrect Reference Values

Our first implementation naively calculated T-score reference values by taking statistics directly from the LMS parameter files:

```python
# WRONG APPROACH - Don't do this!
def calculate_tscore_reference_values_naive(gender_code):
    # Load LMS data
    df = pd.read_csv(f"data/adults_LMS_appendicular_LMI_gender{gender_code}.csv")
    young_adult_data = df[(df["age"] >= 20) & (df["age"] <= 30)]
    
    # Take statistics of the median (mu) values - THIS IS WRONG!
    mu_values = young_adult_data["mu"].values
    mu_peak = np.mean(mu_values)
    sigma_peak = np.std(mu_values, ddof=1)
    
    return mu_peak, sigma_peak
```

This gave us results like:
- **μ = 8.281** (reasonable)  
- **σ = 0.064** (way too small!)

The problem? We were calculating the standard deviation of **median values across different ages**, not the actual **population variation** we needed for T-score calculation.

### Understanding LMS Parameters vs Population Statistics

The LMS method describes population distributions using three age-varying parameters:

- **L (lambda)**: Skewness parameter for Box-Cox transformation
- **M (mu)**: Median of the distribution at that age
- **S (sigma)**: Coefficient of variation at that age

These parameters **describe the shape of the distribution** but aren't directly the population mean and standard deviation we need for T-score calculation.

#### What We Actually Need

For T-score calculation, we need:
- **μ_peak**: True population mean for young adults (ages 20-30)
- **σ_peak**: True population standard deviation for young adults

### The Solution: Monte Carlo Sampling from LMS Distributions

Our solution uses Monte Carlo sampling to generate a synthetic population from the LMS distributions, then calculate empirical statistics:

#### Step 1: Sample Across Young Adult Age Range

```python
def calculate_tscore_reference_values(gender_code):
    # Load ALMI LMS data for the specified gender
    data_file = f"data/adults_LMS_appendicular_LMI_gender{gender_code}.csv"
    df = pd.read_csv(data_file)
    
    # Filter to young adult age range (20-30 years) - peak muscle mass period
    young_adult_data = df[(df["age"] >= 20) & (df["age"] <= 30)]
    
    all_samples = []
    n_samples_per_age = 1000  # Sample size per age year
```

#### Step 2: Generate Samples from Each Age's LMS Distribution

For each age in the 20-30 range, we sample from that age's specific LMS distribution:

```python
    for _, row in young_adult_data.iterrows():
        L, M, S = row["lambda"], row["mu"], row["sigma"]
        
        # Generate samples from the LMS distribution for this age
        # Use percentile sampling approach for robustness
        percentiles = np.linspace(0.01, 0.99, n_samples_per_age)
        z_scores = stats.norm.ppf(percentiles)
        
        # Convert Z-scores to ALMI values using LMS transformation
        samples = []
        for z in z_scores:
            almi_value = get_value_from_zscore(z, L, M, S)
            if not pd.isna(almi_value) and almi_value > 0:
                samples.append(almi_value)
        
        all_samples.extend(samples)
```

#### Step 3: Calculate Empirical Population Statistics

Once we have our simulated population, we calculate the actual statistics:

```python
    # Calculate empirical population statistics
    mu_peak = np.mean(all_samples)
    sigma_peak = np.std(all_samples, ddof=1)  # Sample standard deviation
    
    return mu_peak, sigma_peak
```

### The LMS Inverse Transformation

The key piece that makes this work is the `get_value_from_zscore` function, which performs the inverse LMS transformation:

```python
def get_value_from_zscore(z, l_val, m_val, s_val, eps=1e-5):
    """
    Calculates a metric value from a Z-score (inverse LMS transformation).
    """
    if pd.isna(z) or pd.isna(l_val) or pd.isna(m_val) or pd.isna(s_val):
        return np.nan

    if abs(l_val) < eps:
        # When L is close to zero, use the logarithmic form
        return m_val * np.exp(s_val * z)
    else:
        # Standard inverse Box-Cox transformation
        return m_val * ((l_val * s_val * z + 1) ** (1 / l_val))
```

This function converts a standard normal Z-score back into the original metric space (ALMI kg/m²) using the LMS parameters for that specific age.

## Why This Approach Works

### 1. Representative Sampling
By sampling across the entire young adult age range (20-30), we capture the natural variation in peak muscle mass across these ages.

### 2. Proper Distribution Sampling 
We're not just taking medians - we're sampling the full distribution at each age, including the tails where people with high and low muscle mass exist.

### 3. Percentile-Based Sampling
Using `np.linspace(0.01, 0.99, 1000)` ensures we sample uniformly across the probability space, giving us a representative population.

### 4. Age-Weighted Combination
Since we sample equally from each age (20, 21, 22, ... 30), we effectively create a uniform age distribution across the peak muscle mass years.

## Results: Realistic T-Score Reference Values

The new approach gives us much more realistic reference values:

```python
# Test the results
male_mu, male_sigma = calculate_tscore_reference_values(0)     # Male
female_mu, female_sigma = calculate_tscore_reference_values(1) # Female

print(f"Male reference: μ={male_mu:.3f}, σ={male_sigma:.3f}")
print(f"Female reference: μ={female_mu:.3f}, σ={female_sigma:.3f}")

# Output:
# Male reference: μ=8.351, σ=0.989
# Female reference: μ=6.413, σ=0.809
```

Now our T-score calculations produce clinically meaningful results:

```python
# Example T-score calculations for males
test_almi_values = [7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
for almi in test_almi_values:
    t_score = calculate_t_score(almi, male_mu, male_sigma)
    print(f"ALMI {almi:.1f} → T-score {t_score:.2f}")

# Output:
# ALMI 7.5 → T-score -0.86
# ALMI 8.0 → T-score -0.35  
# ALMI 8.5 → T-score 0.15
# ALMI 9.0 → T-score 0.66
# ALMI 9.5 → T-score 1.16
# ALMI 10.0 → T-score 1.67
```

## Peak Zone Implementation

With proper reference values, we can now implement meaningful peak muscle mass zone stratification:

```python
def get_tscore_peak_zone(t_score):
    """Convert T-score to peak muscle mass zone classification."""
    if t_score >= 2.0:
        return "Elite Zone (T ≥ +2.0)", "#228B22"     # Dark Green
    elif t_score >= 0:
        return "Peak Zone (0 ≤ T < +2.0)", "#90EE90"    # Light Green
    elif t_score >= -1.0:
        return "Approaching Peak (-1.0 ≤ T < 0)", "#FFD700"    # Yellow  
    elif t_score >= -2.0:
        return "Below Peak (-2.0 ≤ T < -1.0)", "#FFA500"  # Orange
    else:
        return "Well Below Peak (T < -2.0)", "#FF6B6B"     # Red
```

### Elite Zone Definition

The **Elite Zone** (T ≥ +2.0) represents exceptional muscle mass - the top ~2.3% of young adults at their peak. This zone:
- Provides motivation for high performers
- Identifies potential athletic/genetic outliers  
- Distinguishes "exceptional" from merely "good" muscle mass
- Offers a meaningful achievement level beyond basic health recommendations

## Performance Considerations

### Sampling Size
We use 1,000 samples per age year, giving us ~11,000 total samples (11 ages × 1,000 samples). This provides:
- **Statistical robustness**: Large enough sample for stable statistics
- **Computational efficiency**: Fast enough for real-time calculation
- **Memory efficiency**: Samples are discarded after statistics calculation

### Caching Opportunity
Since reference values only depend on gender, they could be pre-calculated and cached:

```python
# Could be pre-calculated and stored
TSCORE_REFERENCES = {
    'male': (8.351, 0.989),
    'female': (6.413, 0.809)
}
```

## Validation and Testing

We implemented comprehensive unit tests to ensure our approach works correctly:

```python
class TestTScoreCalculations(unittest.TestCase):
    
    def test_tscore_reference_values_calculation(self):
        """Test that T-score reference values are calculated correctly."""
        male_mu, male_sigma = calculate_tscore_reference_values(0)
        
        # Should return valid numeric values
        self.assertFalse(np.isnan(male_mu))
        self.assertFalse(np.isnan(male_sigma))
        
        # Male ALMI values should be in reasonable range (6-12 kg/m²)
        self.assertGreater(male_mu, 6.0)
        self.assertLess(male_mu, 12.0)
        
        # Standard deviation should be reasonable (0.5-2.0 kg/m²)  
        self.assertGreater(male_sigma, 0.5)
        self.assertLess(male_sigma, 2.0)
    
    def test_tscore_risk_stratification(self):
        """Test T-score risk zone boundaries."""
        male_mu, male_sigma = calculate_tscore_reference_values(0)
        
        # Test that T-scores fall into expected risk zones
        test_values = [
            (male_mu + 2 * male_sigma, "Normal"),      # Well above average
            (male_mu - 0.5 * male_sigma, "Mild Risk"), # Below average
            (male_mu - 1.5 * male_sigma, "Moderate Risk"), # Low
            (male_mu - 2.5 * male_sigma, "Severe Risk"),   # Very low
        ]
        
        for almi_value, expected_risk_zone in test_values:
            t_score = calculate_t_score(almi_value, male_mu, male_sigma)
            # Verify T-score is in expected range...
```

## Conclusion

This implementation demonstrates how to bridge the gap between **statistical parameters** (LMS values) and **clinical interpretation** (T-scores). By using Monte Carlo sampling, we:

1. **Honor the original research**: Use LMS parameters exactly as published
2. **Generate realistic populations**: Sample full distributions, not just medians  
3. **Calculate meaningful statistics**: Get proper population variance for T-scores
4. **Enable clinical interpretation**: Provide T-scores that align with established risk zones

The key insight is that **LMS parameters describe distributions, not populations**. To get population statistics, you need to sample from those distributions and calculate empirical values - exactly what our Monte Carlo approach accomplishes.

### Implementation Files
- [`core.py:273-353`](https://github.com/benmabey/bodymetrics/blob/d1d537b94ccf4fc83ae2db4101758425dee281dd/core.py#L273-L353): New `calculate_tscore_reference_values()` function
- [`tests/unit/test_zscore_calculations.py:2155-2297`](https://github.com/benmabey/bodymetrics/blob/d1d537b94ccf4fc83ae2db4101758425dee281dd/tests/unit/test_zscore_calculations.py#L2155-L2297): Comprehensive T-score test suite
- [`webapp.py:1854-1867, 2008-2021`](https://github.com/benmabey/bodymetrics/blob/d1d537b94ccf4fc83ae2db4101758425dee281dd/webapp.py#L1854-L1867): Web interface integration with dual-mode plotting

This approach now provides the realistic T-score reference values needed for meaningful body composition analysis in the longevity and healthspan optimization context.
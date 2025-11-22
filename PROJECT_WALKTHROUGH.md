# Telco Customer Churn Analysis - Project Walkthrough

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Cleaning](#data-cleaning)
3. [Research Question and Hypothesis](#research-question-and-hypothesis)
4. [Assumption Checks](#assumption-checks)
5. [Model Building](#model-building)
6. [Hypothesis Testing](#hypothesis-testing)
7. [Visualizations](#visualizations)
8. [Results and Interpretations](#results-and-interpretations)
9. [Conclusions](#conclusions)

---

## Project Overview

### Dataset
- **Source**: Telco Customer Churn Dataset
- **Initial Size**: 7,045 rows × 21 columns
- **Final Cleaned Size**: 7,032 rows × 21 columns
- **Target Variable**: Churn (Yes/No)
- **Key Predictors**: Contract Type, Online Security, Tech Support, Monthly Charges

### Project Goals
1. Clean and prepare the dataset for analysis
2. Test hypothesis Q2: Do online security, tech support, and monthly charges add to predicting churn, even with contract type considered?
3. Perform multiple logistic regression with proper assumption checks
4. Create meaningful visualizations
5. Interpret results and draw conclusions

---

## Data Cleaning

### Step 1: Initial Data Inspection
- Loaded dataset with 7,045 observations
- Identified 21 variables including:
  - Customer demographics (gender, SeniorCitizen, Partner, Dependents)
  - Service subscriptions (Phone, Internet, Security, Backup, etc.)
  - Account information (tenure, contract, payment method, charges)
  - Target variable (Churn)

### Step 2: Missing Values Check
- **Finding**: No explicit missing values detected initially
- **Issue Discovered**: `TotalCharges` column contained empty strings (whitespace) instead of NaN
- **Solution**: Converted empty strings to NaN, then converted column to numeric type

### Step 3: Data Type Corrections
- **TotalCharges**: Converted from object to float64
- **Issue**: 11 rows had missing TotalCharges (customers with tenure 0-1 months)
- **Decision**: Dropped 11 rows with missing TotalCharges
- **Final Dataset**: 7,032 rows

### Step 4: Duplicate Check
- **Result**: No duplicate rows found
- All observations are unique customers

### Step 5: Categorical Variable Examination
- Verified all categorical variables have valid values:
  - Contract: Month-to-month, One year, Two year
  - Churn: Yes, No
  - Services: Yes, No, No internet service
- No inconsistencies found

### Step 6: Outlier Detection
- **Method**: IQR (Interquartile Range) method
- **Findings**:
  - Outliers detected in tenure, MonthlyCharges, and TotalCharges
  - **Decision**: Kept all outliers as they represent legitimate customer data:
    - Long-term customers (high tenure)
    - Premium service customers (high charges)
    - These are meaningful for churn prediction

### Step 7: Key Variable Verification
- **Contract Distribution**:
  - Month-to-month: 55.1%
  - One year: 20.9%
  - Two year: 23.9%
- **Churn Distribution**:
  - No: 73.4%
  - Yes: 26.6%

### Step 8: Data Export
- Saved cleaned dataset as `telco_customer_churn_cleaned.csv`
- Ready for analysis

---

## Research Question and Hypothesis

### Research Question Q2
**Do things like online security, tech support, monthly charges add to predicting churn, even with contract type considered?**

### Hypotheses
- **H₀ (Null Hypothesis)**: Online security, tech support, and monthly charges do not significantly improve churn prediction beyond contract type alone.
- **H₁ (Alternative Hypothesis)**: At least one of these features (online security, tech support, monthly charges) significantly improves churn prediction beyond contract type.

### Methodology
We use a **Likelihood Ratio Test** to compare two nested models:
- **Model 1 (Null Model)**: Churn ~ Contract Type
- **Model 2 (Full Model)**: Churn ~ Contract Type + OnlineSecurity + TechSupport + MonthlyCharges

If the full model significantly improves fit, we reject H₀.

---

## Assumption Checks

Before building the models, we verified all key assumptions for logistic regression:

### 1. Sample Size Adequacy ✓ PASSED
- **Total Sample Size**: 7,032 observations
- **Model 1**: 3,516 observations per predictor (well above 10-20 minimum)
- **Model 2**: 1,406 observations per predictor (well above 10-20 minimum)
- **Conclusion**: Sample size is more than adequate for both models

### 2. Multicollinearity (VIF) ✓ PASSED
- **Variance Inflation Factor Results**:
  - Contract_One year: 1.16
  - Contract_Two year: 1.28
  - OnlineSecurity_binary: 1.25
  - TechSupport_binary: 1.33
  - MonthlyCharges_scaled: 1.24
- **Max VIF**: 1.33 (well below threshold of 5)
- **Conclusion**: No multicollinearity issues detected

### 3. Linearity of Logit ✓ PASSED
- **Method**: Binned MonthlyCharges into 10 quantiles and calculated logit for each
- **Correlation**: 0.793 between MonthlyCharges and logit
- **Interpretation**: Strong linear relationship (above 0.7 threshold)
- **Conclusion**: Assumption satisfied

### 4. No Perfect Separation ✓ PASSED
- **Check**: Examined cross-tabulations for each binary predictor
- **Results**: All predictors show churn rates between 2.8% and 34.1%
- **Finding**: No perfect separation (no 0% or 100% churn rates)
- **Conclusion**: No convergence issues expected

### 5. Independence of Observations ✓ PASSED
- **Data Type**: Cross-sectional dataset
- **Structure**: Each row represents a unique customer
- **Conclusion**: Observations are independent (no repeated measures or clustering)

### Overall Assessment
**All assumptions are satisfied.** The logistic regression models are valid and reliable.

---

## Model Building

### Data Preparation
1. **Target Variable**: Converted Churn from Yes/No to binary (1/0)
2. **Contract Type**: Created dummy variables (Month-to-month as reference)
3. **Online Security**: Converted to binary (Yes=1, No=0)
4. **Tech Support**: Converted to binary (Yes=1, No=0)
5. **Monthly Charges**: Standardized using StandardScaler

### Model 1: Null Model
**Formula**: Churn ~ Contract Type

**Predictors**:
- Contract_One year (reference: Month-to-month)
- Contract_Two year (reference: Month-to-month)

**Results**:
- **Log-Likelihood**: -3,381.26
- **AIC**: 6,768.52
- **Pseudo R-squared**: 0.1696
- **Converged**: Yes

**Coefficients**:
- Contract_One year: -1.769 (p < 0.001)
- Contract_Two year: -3.236 (p < 0.001)

**Interpretation**: Both one-year and two-year contracts significantly reduce churn compared to month-to-month contracts.

### Model 2: Full Model
**Formula**: Churn ~ Contract Type + OnlineSecurity + TechSupport + MonthlyCharges

**Predictors**:
- Contract_One year
- Contract_Two year
- OnlineSecurity_binary
- TechSupport_binary
- MonthlyCharges_scaled

**Results**:
- **Log-Likelihood**: -3,162.29
- **AIC**: 6,336.58
- **Pseudo R-squared**: 0.2233
- **Converged**: Yes

**Coefficients**:
- Contract_One year: -1.625 (p < 0.001)
- Contract_Two year: -2.865 (p < 0.001)
- OnlineSecurity_binary: -0.726 (p < 0.001)
- TechSupport_binary: -0.589 (p < 0.001)
- MonthlyCharges_scaled: 0.666 (p < 0.001)

**Interpretation**:
- Longer contracts reduce churn
- Online Security reduces churn
- Tech Support reduces churn
- Higher Monthly Charges increase churn

### Model 3: Decision Tree Classifier
**Purpose**: To provide a non-linear classification model and fulfill the requirement of applying at least 3 modeling methods.

**Configuration**:
- Max Depth: 5 (to prevent overfitting)
- Split: 80% Train, 20% Test

**Results**:
- **Accuracy**: 78.82%
- **Precision (Churn=0)**: 0.82
- **Recall (Churn=0)**: 0.91
- **Precision (Churn=1)**: 0.64
- **Recall (Churn=1)**: 0.46

**Key Findings**:
- The model achieves ~79% accuracy, which is comparable to the logistic regression models.
- It identifies key predictors similar to the logistic regression (Contract, Charges, Tenure).

---

## Hypothesis Testing

### Likelihood Ratio Test

**Test Statistic**: LR = -2 × (LL_null - LL_full)

**Results**:
- **Null Model Log-Likelihood**: -3,381.26
- **Full Model Log-Likelihood**: -3,162.29
- **LR Test Statistic**: 437.94
- **Degrees of Freedom**: 3 (OnlineSecurity, TechSupport, MonthlyCharges)
- **Critical Value (α=0.05)**: 7.81
- **P-value**: < 0.001

### Decision
**✓ REJECT H₀**

The full model significantly improves churn prediction. At least one of the additional features (OnlineSecurity, TechSupport, MonthlyCharges) significantly contributes to predicting churn beyond contract type alone.

### Model Comparison

| Metric | Model 1 (Null) | Model 2 (Full) | Improvement |
|--------|----------------|----------------|-------------|
| Log-Likelihood | -3,381.26 | -3,162.29 | Higher (better) |
| AIC | 6,768.52 | 6,336.58 | Lower (better) |
| BIC | 6,789.20 | 6,378.20 | Lower (better) |
| Pseudo R-squared | 0.1696 | 0.2233 | Higher (better) |

**Conclusion**: Model 2 (Full Model) is superior in all metrics.

---

## Visualizations

### Visualization 1: Churn Rates by Key Features
**Purpose**: Show how churn varies across different feature combinations

**Components** (2×2 grid):
1. **Churn Rate by Contract Type**
   - Month-to-month: Highest churn rate
   - One year: Moderate churn rate
   - Two year: Lowest churn rate

2. **Churn Rate by Online Security**
   - No Security: Higher churn
   - Has Security: Lower churn

3. **Churn Rate by Tech Support**
   - No Support: Higher churn
   - Has Support: Lower churn

4. **Churn Rate by Monthly Charges** (binned)
   - Low charges: Lower churn
   - High charges: Higher churn

**Key Observations**:
- Month-to-month contracts have the highest churn rate
- Customers without online security have higher churn
- Customers without tech support have higher churn
- Higher monthly charges are associated with higher churn rates

### Visualization 2: Interaction Effects
**Purpose**: Show how additional features interact with contract type

**Components** (3 panels):
1. **Contract Type × Online Security**
   - Security reduces churn across all contract types
   - Effect is consistent regardless of contract length

2. **Contract Type × Tech Support**
   - Tech support reduces churn across all contract types
   - Effect is consistent regardless of contract length

3. **Contract Type × Monthly Charges**
   - Higher charges increase churn, especially for month-to-month contracts
   - Two-year contracts show lower churn even at higher charge levels

**Key Observations**:
- Online Security reduces churn across all contract types
- Tech Support reduces churn across all contract types
- Higher monthly charges increase churn, especially for month-to-month contracts
- The protective effect of security/support is consistent across contract types

### Visualization 3: Model Comparison - Odds Ratios
**Purpose**: Visualize coefficients and compare models

**Components** (2 panels):
1. **Model 1 Odds Ratios**: Contract Type only
2. **Model 2 Odds Ratios**: Contract + Additional Features

**Odds Ratio Interpretation**:
- **OR < 1**: Decreases odds of churn (protective factor)
- **OR > 1**: Increases odds of churn (risk factor)
- **OR = 1**: No effect

**Model 2 Key Findings**:
- One year contract: OR = 0.20 (reduces churn by 80%)
- Two year contract: OR = 0.06 (reduces churn by 94%)
- Online Security: OR = 0.48 (reduces churn by 52%)
- Tech Support: OR = 0.55 (reduces churn by 45%)
- Monthly Charges (scaled): OR = 1.95 (increases churn by 95% per SD)

---

## Results and Interpretations

### Statistical Findings

#### 1. Hypothesis Test Result
- **LR Statistic**: 437.94
- **P-value**: < 0.001
- **Conclusion**: **REJECT H₀** at α = 0.05
- The additional features significantly improve churn prediction

#### 2. Individual Feature Significance (from Model 2)

| Feature | Coefficient | P-value | Significant? |
|---------|-------------|---------|--------------|
| Contract_One year | -1.625 | < 0.001 | Yes |
| Contract_Two year | -2.865 | < 0.001 | Yes |
| OnlineSecurity_binary | -0.726 | < 0.001 | Yes |
| TechSupport_binary | -0.589 | < 0.001 | Yes |
| MonthlyCharges_scaled | 0.666 | < 0.001 | Yes |

**All features are statistically significant** (p < 0.001).

#### 3. Key Patterns and Relationships
- **Contract Type**: Longer contracts (One year, Two year) significantly reduce churn
- **Online Security**: Having online security significantly reduces churn
- **Tech Support**: Having tech support significantly reduces churn
- **Monthly Charges**: Higher charges significantly increase churn

#### 4. Practical Implications
- **Contract Strategy**: Companies should encourage longer-term contracts to reduce churn
- **Service Promotion**: Promoting online security and tech support can help retain customers
- **Pricing Strategy**: Pricing should balance revenue with churn risk
- **Model Performance**: The full model provides better predictive power than contract type alone

### Summary Statistics

| Variable | Value |
|----------|-------|
| Total Sample Size | 7,032 |
| Churn Rate | 26.57% |
| Monthly Charges - Mean | $64.76 |
| Monthly Charges - Std Dev | $30.09 |
| Monthly Charges - Min | $18.25 |
| Monthly Charges - Max | $118.75 |
| Has Online Security | 28.64% |
| Has Tech Support | 29.01% |
| Month-to-month Contract | 55.05% |
| One year Contract | 20.93% |
| Two year Contract | 23.99% |
| Tenure - Mean | 32.37 months |
| Tenure - Std Dev | 24.29 months |

---

## Conclusions

### Research Question Answer
**Yes, online security, tech support, and monthly charges significantly add to predicting churn beyond contract type alone.**

### Key Findings

1. **All Additional Features Matter**
   - Online Security, Tech Support, and Monthly Charges all significantly improve churn prediction
   - Each feature is statistically significant (p < 0.001)

2. **Model Improvement**
   - Full model (Model 2) significantly outperforms null model (Model 1)
   - Pseudo R-squared increased from 0.17 to 0.22
   - AIC decreased from 6,769 to 6,337

3. **Protective Factors**
   - Longer contracts (especially two-year) strongly reduce churn
   - Online Security reduces churn by ~52%
   - Tech Support reduces churn by ~45%

4. **Risk Factors**
   - Higher monthly charges increase churn risk
   - Month-to-month contracts have highest churn risk

5. **Assumption Validity**
   - All logistic regression assumptions are satisfied
   - Results are reliable and valid
   - Hypothesis test conclusions are trustworthy

### Business Recommendations

1. **Encourage Long-term Contracts**: Offer incentives for one-year and two-year contracts
2. **Promote Value-added Services**: Market online security and tech support as churn-reduction benefits
3. **Pricing Strategy**: Balance premium pricing with churn risk, especially for month-to-month customers
4. **Targeted Retention**: Focus retention efforts on month-to-month customers with high charges and no security/support

### Statistical Reliability

- ✅ All assumptions satisfied
- ✅ Large sample size (7,032 observations)
- ✅ No multicollinearity issues
- ✅ Strong model fit
- ✅ Highly significant results (p < 0.001)
- ✅ Robust findings across multiple metrics

**The hypothesis test results are reliable and valid. The rejection of H₀ is well-supported by the data.**

---

## Files Created

1. **data_cleaning.ipynb**: Comprehensive data cleaning workflow
2. **multiple_logistic_regression_hypothesis_test.ipynb**: Hypothesis testing and analysis
3. **telco_customer_churn_cleaned.csv**: Cleaned dataset ready for analysis
4. **PROJECT_WALKTHROUGH.md**: This comprehensive walkthrough document
5. **decision_tree_model.ipynb**: Decision Tree Classifier implementation

---

## Technical Details

### Libraries Used
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib & seaborn: Visualizations
- scipy: Statistical tests
- statsmodels: Logistic regression modeling
- sklearn: Data preprocessing and Decision Tree

### Statistical Methods
- Logistic Regression
- Decision Tree Classifier
- Likelihood Ratio Test
- Variance Inflation Factor (VIF)
- Odds Ratio interpretation
- Model comparison metrics (AIC, BIC, Pseudo R-squared)

---

*This walkthrough documents the complete analysis workflow from data cleaning through hypothesis testing and interpretation.*


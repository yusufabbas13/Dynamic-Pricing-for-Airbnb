# Dynamic-Pricing-for-Airbnb
 Machine learning pipeline that predicts the **relative price tier** (Low / Medium / High) for Airbnb listings on a monthly basis using air roi temporal data. The model helps hosts and platforms make **data-driven pricing decisions** by classifying where a listing stands relative to its local city market.

## Pipeline Summary
```
Raw Airbnb CSV
      │
      ▼
Column Selection & Surcharge Engineering
      │
      ▼
Missing Value Imputation (city-group medians)
      │
      ▼
Categorical Encoding
(City → one-hot, cancellation_policy → ordinal, num_reviews → binned)
      │
      ▼
Amenity Selection (10–40% prevalence filter)
      │
      ▼
Feature Engineering
(lag features, cyclical month encoding, market comparison, quality score)
      │
      ▼
Target Creation
(city-relative price tier: Low=0 / Medium=1 / High=2)
      │
      ▼
Temporal Train / Test Split (75% / 25% by date)
      │
      ▼
XGBoost Multiclass Classifier (balanced class weights)
      │
      ▼
Evaluation + SHAP Explainability
      │
      ▼
predictions.csv
```

---

## Features

### Lag & Temporal Features
| Feature | Description |
|---|---|
| `price_lag1` | Listing's own price from the previous month |
| `occupancy_lag1` | Listing's occupancy from the previous month |
| `month_sin` / `month_cos` | Cyclical encoding of month (captures seasonality) |

### Demand & Pressure Features
| Feature | Description |
|---|---|
| `pressure_t` | `reserved_days / (vacant_days + 1)` — booking pressure |
| `occupancy_change` | Month-over-month change in occupancy |
| `urgency` | `1 / (booking_lead_time_avg + 1)` when reserved — last-minute booking signal |

### Market Comparison Features (Leave-One-Out)
| Feature | Description |
|---|---|
| `city_price_mean` | City-date average price excluding current listing |
| `city_price_median` | City-date median price |
| `city_occupancy_mean` | City-date average occupancy excluding current listing |
| `price_vs_city_mean` | `price_lag1 / city_price_mean` — relative positioning |
| `occupancy_vs_city_mean` | `occupancy_lag1 / city_occupancy_mean` |

### Quality & Host Features
| Feature | Description |
|---|---|
| `quality_score` | `rating_overall × log(num_reviews_bin + 1)` |
| `guest_type_proxy` | `length_of_stay_avg / (booking_lead_time_avg + 1)` |
| `surchages` | Combined cleaning fee + extra guest fee |
| `superhost` | Binary superhost flag |
| `professional_management` | Binary management flag |
| `instant_book` | Binary instant booking flag |

### One-Hot Features
- `city_*` — City dummies
- Selected amenities (10–40% prevalence across listings)

### Target
| Label | Meaning |
|---|---|
| `0` — Low | Bottom third of prices in that city-month |
| `1` — Medium | Middle third |
| `2` — High | Top third |

---

## Model

**XGBoost Multiclass Classifier**
```python
XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=1,
    reg_alpha=1,
    reg_lambda=1,
    random_state=42
)
```

**Class balancing:** Balanced class weights computed via `sklearn.utils.class_weight.compute_class_weight` and passed as `sample_weight` during fitting.

**Train/Test Split:** Temporal — first 75% of dates for training, most recent 25% for testing.

---

## Results

Evaluated on the held-out most recent 25% of dates:

| Metric | Score |
|---|---|
| Accuracy | *run notebook to generate* |
| Macro F1 | *run notebook to generate* |
| Weighted F1 | *run notebook to generate* |

Per-class breakdown (Low / Medium / High) available via `classification_report` in the notebook.

SHAP summary plots saved as:
- `shap_class_0_Low_Price.png`
- `shap_class_1_Medium_Price.png`
- `shap_class_2_High_Price.png`

> Metrics are left blank — run the notebook on your own data snapshot to generate current numbers.

---

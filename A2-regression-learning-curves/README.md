# A2: The Morning Rush

Hourly bike-rental forecasting for the 30 busiest Taipei YouBike 2.0 stations
(Jan–Jun 2026; 108,720 station-hours for training, 21,600 for testing), using linear models only.

## Assignment overview

The assignment builds an end-to-end regression pipeline in three parts: least squares by hand,
feature engineering and scaling, and learning-curve diagnosis with a Kaggle leaderboard submission.

**Part I: Least squares by hand**
- Derive the ridge solution and show it is unique for every λ > 0
- OLS and ridge on a five-point precipitation vs. rentals example; the slope as λ → ∞
- Unit changes (mm → cm): OLS predictions are invariant, ridge is not; why gradient descent needs scaled features

**Part II: Feature engineering and scaling**
- Explore demand over time, across stations and with the weather; compare the weather-only baseline with a log(1 + y) target
- Five cumulative feature sets with a fixed Ridge model: one-hot hour and weekday, workday flag,
  station identity, and crossed station × hour × workday indicators
- Compare no scaling, StandardScaler, MinMaxScaler and RobustScaler for LinearRegression, Ridge and SGDRegressor

**Part III: Learning curves and forecasting**
- Diagnose a high-bias model (weather + numeric hour) and a high-variance model (degree-2 polynomial, no regularisation)
- Remedies: richer features for the biased model, a tuned ridge penalty for the overfitting model
- Select the final model on the May validation set, retrain on January–May and predict June

## Contents

| File | Description |
|---|---|
| `A2_315511079.pdf` | Report (Part I derivations, results, analysis) |
| `A2_315511079.ipynb` | Part II and Part III: exploration, feature engineering, scaling, learning curves, final model and submission |
| `kaggle_submission.csv` | Final test-set predictions submitted to the Kaggle leaderboard (21,600 rows, test RMSE 11.68) |
| `ML-part2-figure/`, `ML-part3-figure/` | Exported figures and result tables |

## Key results

- Weather-only baseline: validation RMSE 29.37 (R² 0.10); best feature set (station × hour × workday, Ridge on log(1 + y)): validation RMSE 13.42 (R² 0.81)
- Scaling leaves OLS unchanged, changes Ridge, and decides whether SGD converges (diverged unscaled; 17–18 epochs scaled)
- Model A has high bias (train 27.0 / CV 27.6); Model B has high variance (train 10.6 / CV 172); both fixed by the remedies
- Kaggle test RMSE **11.68** vs. TA baseline 24.25

## Run

Open the notebook in Google Colab
([link](https://colab.research.google.com/drive/14agXEi8Io2W_C5021Mbf59dkZ5SfQX3I?usp=sharing)),
upload `train.csv`, `test.csv` and `sample_submission.csv` from the course Kaggle competition,
and run all cells. Random seeds are fixed to 0.

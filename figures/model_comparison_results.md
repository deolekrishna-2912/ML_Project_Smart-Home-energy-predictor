# Model Comparison Results — Smart Home Appliance Energy Predictor

**Split strategy:** random
**Evaluation set:** Test split (4 models evaluated)

| # | Model | MAE (Wh) | RMSE (Wh) | R² |
|---|-------|----------|-----------|------|
| 1 | Linear Regression | 52.55 | 91.17 | 0.1694 |
| 2 | Decision Tree | 42.37 | 84.35 | 0.2890 |
| 3 | Random Forest | 35.63 | 73.27 | 0.4636 |
| 4 | HistGradientBoosting 🏆 | 35.42 | 70.47 | 0.5037 |

> **Best model:** HistGradientBoosting (lowest RMSE)
>
> Serialised pipeline saved to `models/best_appliance_model.joblib`

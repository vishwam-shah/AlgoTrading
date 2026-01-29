# Pipeline Model, Factors, Features, and Logging: Full Transparency

## 1. Models Used in the Pipeline
- **XGBoost Classifier**: Predicts next-day (T+1) direction (up/down) for each stock.
- **LightGBM Classifier**: Also predicts T+1 direction for each stock.
- **Ensemble Model**: Combines XGBoost and LightGBM predictions (average probability, majority vote) for more robust direction prediction.
- **XGBoost Regressor**: Predicts expected return for each stock.
- **LightGBM Quantile Regressors**: Predict upper/lower bounds for returns (risk estimation).
- **Optional Deep Learning Models**: LSTM/GRU for sequence-based prediction (not always enabled in production pipeline).

## 2. What Each Model Predicts
- **Direction Classifiers**: Output probability and binary prediction (up/down) for next day.
- **Regressors**: Output expected return (numeric value) for next day.
- **Quantile Models**: Output upper/lower bounds for risk management.
- **Ensemble**: Combines classifier outputs for final direction prediction (used for signals and portfolio decisions).

## 3. Ensemble Logic
- **Ensemble Direction**: If average probability > 0.5, predict 'up'; else 'down'.
- **Ensemble Accuracy**: Used as main metric for pipeline performance (shown as Direction Acc in frontend).

## 4. Factors and Their Impact
- **Value**: Measures undervaluation (low PE, high book value). Higher value score increases stock weight.
- **Momentum**: Measures recent price strength. Higher momentum increases allocation.
- **Quality**: Measures profitability, stability. Higher quality increases allocation.
- **Low Volatility**: Favors stocks with stable returns. Higher score increases allocation.
- **Sentiment**: Measures news sentiment. Positive sentiment increases allocation.
- **Combined Score**: Weighted sum of all factors, used for ranking and selection.
- **Impact Logging**: Each factor's score per stock is logged in pipeline stepwise CSVs for transparency.

## 5. Feature Count: Why Reduced?
- **Earlier**: 388+ features (all raw, engineered, technical, sentiment, macro).
- **Now**: ~118 features (after feature selection, redundancy removal, and performance tuning).
- **Reason**: Fewer features = faster training, less overfitting, more robust models. Only top features (by importance, correlation, and stability) are kept.
- **Feature Selection**: Automated via model importance, correlation analysis, and manual review. Full feature list logged in CSV for each run.

## 6. Logging and Auditability
- **Stepwise CSVs**: Every pipeline step logs results in CSV (data, features, factors, model metrics, backtest, signals) in `results/pipeline_runs/{job_id}/`.
- **Parameter Sweeps**: For every change (feature count, sample size, prediction horizon), a separate run is logged with all metrics and results.
- **Change Tracking**: Each run is timestamped and job_id is used for traceability.
- **Improvement Suggestions**: Always compare metrics (direction accuracy, return, Sharpe, drawdown) across runs to decide if changes help.

## 7. How to Improve
- **Feature Engineering**: Try adding/removing features, log impact on accuracy and returns.
- **Model Tuning**: Adjust hyperparameters, try more/less complex models, log results.
- **Factor Weighting**: Experiment with different factor weights, log portfolio changes.
- **Prediction Horizon**: Try T+1, T+5, T+10 predictions, log results.
- **Sample Size**: Use more/less data, log impact.
- **Portfolio Optimization**: Test different methods (risk parity, max Sharpe, min vol), log allocations and performance.

## 8. Next Steps: Parameter Sweep Scripts
- Create scripts to run pipeline with different parameters (feature count, sample size, prediction horizon, factor weights).
- Each script outputs detailed CSVs for every run, enabling full audit trail and comparison.
- Use these logs to decide which changes improve the system and which do not.

---

**No sugarcoating:**
- Direction accuracy is the main metric, but can be misleading if market regime changes.
- Fewer features may reduce overfitting but can miss subtle signals.
- Ensemble is only as good as its base models; if both are weak, ensemble will be weak.
- Sentiment is noisy and should be used cautiously.
- Always log and compare every change; do not trust single runs.

**For full transparency, see the CSVs in `results/pipeline_runs/{job_id}/` and use the parameter sweep scripts to generate more detailed logs.**

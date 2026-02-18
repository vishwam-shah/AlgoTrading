"""
Model-Wise Performance Analysis per Stock
Shows directional accuracy for XGBoost, LightGBM, and Ensemble separately
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv(r'C:\Users\Home\Documents\AI_IN_STOCK_V2\V3\results\all_stocks_summary.csv')

print("="*160)
print(" "*55 + "MODEL-WISE PERFORMANCE ANALYSIS PER STOCK")
print("="*160)

# Calculate combined win rate
df['win_rate_combined'] = (df['win_rate_long'] + df['win_rate_short']) / 2

# Group by stock and model
stock_model_summary = df.groupby(['symbol', 'model']).agg({
    'dir_accuracy': 'mean',
    'win_rate_combined': 'mean',
    'sharpe': 'mean',
    'profit_factor': 'mean',
    'rmse': 'mean',
    'mae': 'mean',
    'n_val': 'mean',
}).reset_index()

# Pivot to show models side by side for directional accuracy
dir_acc_pivot = stock_model_summary.pivot(index='symbol', columns='model', values='dir_accuracy')
dir_acc_pivot.columns = [f'dir_acc_{col}' for col in dir_acc_pivot.columns]

# Pivot for combined win rate
win_rate_pivot = stock_model_summary.pivot(index='symbol', columns='model', values='win_rate_combined')
win_rate_pivot.columns = [f'win_rate_{col}' for col in win_rate_pivot.columns]

# Pivot for profit factor
pf_pivot = stock_model_summary.pivot(index='symbol', columns='model', values='profit_factor')
pf_pivot.columns = [f'pf_{col}' for col in pf_pivot.columns]

# Combine all pivots
combined = pd.concat([dir_acc_pivot, win_rate_pivot, pf_pivot], axis=1).reset_index()

print("\n📊 DIRECTIONAL ACCURACY BY MODEL PER STOCK:\n")
print(f"{'Stock':12s} {'XGB':>10s} {'LGB':>10s} {'Ensemble':>10s} {'Stock Avg':>10s}")
print("─"*160)

for _, row in combined.iterrows():
    stock_avg = (row['dir_acc_xgb'] + row['dir_acc_lgb'] + row['dir_acc_ensemble']) / 3
    print(f"{row['symbol']:12s} "
          f"{row['dir_acc_xgb']:10.4f} "
          f"{row['dir_acc_lgb']:10.4f} "
          f"{row['dir_acc_ensemble']:10.4f} "
          f"{stock_avg:10.4f}")

# Calculate model averages across all stocks
print("─"*160)
xgb_avg = combined['dir_acc_xgb'].mean()
lgb_avg = combined['dir_acc_lgb'].mean()
ens_avg = combined['dir_acc_ensemble'].mean()
overall_avg = (xgb_avg + lgb_avg + ens_avg) / 3

print(f"{'MODEL AVG':12s} "
      f"{xgb_avg:10.4f} "
      f"{lgb_avg:10.4f} "
      f"{ens_avg:10.4f} "
      f"{overall_avg:10.4f}")

print("\n" + "="*160)
print("📊 COMBINED WIN RATE (Long + Short) BY MODEL PER STOCK:\n")
print(f"{'Stock':12s} {'XGB':>10s} {'LGB':>10s} {'Ensemble':>10s} {'Stock Avg':>10s}")
print("─"*160)

for _, row in combined.iterrows():
    stock_avg = (row['win_rate_xgb'] + row['win_rate_lgb'] + row['win_rate_ensemble']) / 3
    print(f"{row['symbol']:12s} "
          f"{row['win_rate_xgb']:10.4f} "
          f"{row['win_rate_lgb']:10.4f} "
          f"{row['win_rate_ensemble']:10.4f} "
          f"{stock_avg:10.4f}")

print("─"*160)
xgb_wr_avg = combined['win_rate_xgb'].mean()
lgb_wr_avg = combined['win_rate_lgb'].mean()
ens_wr_avg = combined['win_rate_ensemble'].mean()
overall_wr_avg = (xgb_wr_avg + lgb_wr_avg + ens_wr_avg) / 3

print(f"{'MODEL AVG':12s} "
      f"{xgb_wr_avg:10.4f} "
      f"{lgb_wr_avg:10.4f} "
      f"{ens_wr_avg:10.4f} "
      f"{overall_wr_avg:10.4f}")

print("\n" + "="*160)
print("📊 PROFIT FACTOR BY MODEL PER STOCK:\n")
print(f"{'Stock':12s} {'XGB':>10s} {'LGB':>10s} {'Ensemble':>10s} {'Stock Avg':>10s}")
print("─"*160)

for _, row in combined.iterrows():
    stock_avg = (row['pf_xgb'] + row['pf_lgb'] + row['pf_ensemble']) / 3
    print(f"{row['symbol']:12s} "
          f"{row['pf_xgb']:10.4f} "
          f"{row['pf_lgb']:10.4f} "
          f"{row['pf_ensemble']:10.4f} "
          f"{stock_avg:10.4f}")

print("─"*160)
xgb_pf_avg = combined['pf_xgb'].mean()
lgb_pf_avg = combined['pf_lgb'].mean()
ens_pf_avg = combined['pf_ensemble'].mean()
overall_pf_avg = (xgb_pf_avg + lgb_pf_avg + ens_pf_avg) / 3

print(f"{'MODEL AVG':12s} "
      f"{xgb_pf_avg:10.4f} "
      f"{lgb_pf_avg:10.4f} "
      f"{ens_pf_avg:10.4f} "
      f"{overall_pf_avg:10.4f}")

print("\n" + "="*160)
print("📈 MODEL PERFORMANCE SUMMARY (Averaged Across All Stocks):")
print("="*160)

summary_data = {
    'Model': ['XGBoost', 'LightGBM', 'Ensemble'],
    'Dir_Accuracy': [xgb_avg, lgb_avg, ens_avg],
    'Win_Rate_Combined': [xgb_wr_avg, lgb_wr_avg, ens_wr_avg],
    'Profit_Factor': [xgb_pf_avg, lgb_pf_avg, ens_pf_avg],
}

summary_df = pd.DataFrame(summary_data)

print(f"\n{'Model':15s} {'Dir.Accuracy':>15s} {'Win Rate (Comb)':>18s} {'Profit Factor':>15s} {'PF Interpretation':>30s}")
print("─"*160)

for _, row in summary_df.iterrows():
    pf_interp = f"₹{row['Profit_Factor']:.2f} per ₹1 lost"
    print(f"{row['Model']:15s} "
          f"{row['Dir_Accuracy']:15.4f} "
          f"{row['Win_Rate_Combined']:18.4f} "
          f"{row['Profit_Factor']:15.4f} "
          f"{pf_interp:>30s}")

# Determine best model
print("\n" + "="*160)
print("🏆 BEST MODEL BY METRIC:")
print("="*160)

best_dir_acc = summary_df.loc[summary_df['Dir_Accuracy'].idxmax()]
best_wr = summary_df.loc[summary_df['Win_Rate_Combined'].idxmax()]
best_pf = summary_df.loc[summary_df['Profit_Factor'].idxmax()]

print(f"  • Best Directional Accuracy: {best_dir_acc['Model']} ({best_dir_acc['Dir_Accuracy']:.4f})")
print(f"  • Best Combined Win Rate:    {best_wr['Model']} ({best_wr['Win_Rate_Combined']:.4f})")
print(f"  • Best Profit Factor:        {best_pf['Model']} ({best_pf['Profit_Factor']:.4f})")

# Save detailed model-wise summary
combined.to_csv(r'C:\Users\Home\Documents\AI_IN_STOCK_V2\V3\results\model_wise_summary.csv', index=False)
summary_df.to_csv(r'C:\Users\Home\Documents\AI_IN_STOCK_V2\V3\results\model_averages.csv', index=False)

print(f"\n💾 Saved model-wise summary to: V3/results/model_wise_summary.csv")
print(f"💾 Saved model averages to: V3/results/model_averages.csv")

print("\n" + "="*160)
print("✅ MODEL-WISE ANALYSIS COMPLETE")
print("="*160)

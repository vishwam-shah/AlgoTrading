"""
Full analysis with totals, averages, and finals for every column.
Produces one master CSV with all detail rows + summary rows.
"""

import pandas as pd
import numpy as np
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
df = pd.read_csv(os.path.join(RESULTS_DIR, 'all_stocks_summary.csv'))

METRICS = ['dir_accuracy', 'win_rate_long', 'win_rate_short', 'sharpe', 'profit_factor', 'rmse', 'mae']

def agg_row(subset, label_col, label_val, extra=None):
    """Build a summary row from a subset."""
    row = {label_col: label_val}
    if extra:
        row.update(extra)
    for m in METRICS:
        row[f'{m}_mean'] = round(subset[m].mean(), 4)
        row[f'{m}_std'] = round(subset[m].std(), 4)
        row[f'{m}_min'] = round(subset[m].min(), 4)
        row[f'{m}_max'] = round(subset[m].max(), 4)
        row[f'{m}_median'] = round(subset[m].median(), 4)
    row['n_runs'] = len(subset)
    row['pct_acc_above_50'] = round((subset['dir_accuracy'] > 0.50).mean() * 100, 1)
    row['pct_sharpe_positive'] = round((subset['sharpe'] > 0).mean() * 100, 1)
    row['pct_profitable'] = round((subset['profit_factor'] > 1.0).mean() * 100, 1)
    return row


# ============================================================================
# 1. PER MODEL PERFORMANCE (across all stocks, all windows)
# ============================================================================
model_rows = []
for model in ['xgb', 'lgb', 'ensemble']:
    sub = df[df['model'] == model]
    row = agg_row(sub, 'model', model)
    # Add per-window breakdown
    for w in df['window'].unique():
        ws = sub[sub['window'] == w]
        row[f'acc_{w}'] = round(ws['dir_accuracy'].mean(), 4)
        row[f'sharpe_{w}'] = round(ws['sharpe'].mean(), 4)
    model_rows.append(row)

# Grand total row
row = agg_row(df, 'model', 'ALL_MODELS_TOTAL')
for w in df['window'].unique():
    ws = df[df['window'] == w]
    row[f'acc_{w}'] = round(ws['dir_accuracy'].mean(), 4)
    row[f'sharpe_{w}'] = round(ws['sharpe'].mean(), 4)
model_rows.append(row)

model_df = pd.DataFrame(model_rows)
model_df.to_csv(os.path.join(RESULTS_DIR, 'final_model_performance.csv'), index=False)
print(f"Saved final_model_performance.csv")

# ============================================================================
# 2. PER STOCK PERFORMANCE (across all windows, all models)
# ============================================================================
stock_rows = []
for symbol in df['symbol'].unique():
    sub = df[df['symbol'] == symbol]
    sector = 'Banking' if symbol in ['HDFCBANK','ICICIBANK','KOTAKBANK','SBIN','AXISBANK'] else 'IT'
    row = agg_row(sub, 'symbol', symbol, {'sector': sector})
    # Per-model breakdown
    for model in ['xgb', 'lgb', 'ensemble']:
        ms = sub[sub['model'] == model]
        row[f'acc_{model}'] = round(ms['dir_accuracy'].mean(), 4)
        row[f'sharpe_{model}'] = round(ms['sharpe'].mean(), 4)
        row[f'pf_{model}'] = round(ms['profit_factor'].mean(), 4)
    # Per-window breakdown
    for w in df['window'].unique():
        ws = sub[sub['window'] == w]
        row[f'acc_{w}'] = round(ws['dir_accuracy'].mean(), 4)
        row[f'sharpe_{w}'] = round(ws['sharpe'].mean(), 4)
    # Best combo
    best_idx = sub['dir_accuracy'].idxmax()
    row['best_combo'] = f"{sub.loc[best_idx, 'model']}_{sub.loc[best_idx, 'window']}"
    row['best_combo_acc'] = sub.loc[best_idx, 'dir_accuracy']
    row['best_combo_sharpe'] = sub.loc[best_idx, 'sharpe']
    stock_rows.append(row)

# Banking total
banking = df[df['symbol'].isin(['HDFCBANK','ICICIBANK','KOTAKBANK','SBIN','AXISBANK'])]
row = agg_row(banking, 'symbol', 'BANKING_TOTAL', {'sector': 'Banking'})
for model in ['xgb', 'lgb', 'ensemble']:
    ms = banking[banking['model'] == model]
    row[f'acc_{model}'] = round(ms['dir_accuracy'].mean(), 4)
    row[f'sharpe_{model}'] = round(ms['sharpe'].mean(), 4)
    row[f'pf_{model}'] = round(ms['profit_factor'].mean(), 4)
for w in df['window'].unique():
    ws = banking[banking['window'] == w]
    row[f'acc_{w}'] = round(ws['dir_accuracy'].mean(), 4)
    row[f'sharpe_{w}'] = round(ws['sharpe'].mean(), 4)
row['best_combo'] = ''
row['best_combo_acc'] = ''
row['best_combo_sharpe'] = ''
stock_rows.append(row)

# IT total
it = df[df['symbol'].isin(['TCS','INFY','HCLTECH','WIPRO','TECHM'])]
row = agg_row(it, 'symbol', 'IT_TOTAL', {'sector': 'IT'})
for model in ['xgb', 'lgb', 'ensemble']:
    ms = it[it['model'] == model]
    row[f'acc_{model}'] = round(ms['dir_accuracy'].mean(), 4)
    row[f'sharpe_{model}'] = round(ms['sharpe'].mean(), 4)
    row[f'pf_{model}'] = round(ms['profit_factor'].mean(), 4)
for w in df['window'].unique():
    ws = it[it['window'] == w]
    row[f'acc_{w}'] = round(ws['dir_accuracy'].mean(), 4)
    row[f'sharpe_{w}'] = round(ws['sharpe'].mean(), 4)
row['best_combo'] = ''
row['best_combo_acc'] = ''
row['best_combo_sharpe'] = ''
stock_rows.append(row)

# Grand total
row = agg_row(df, 'symbol', 'GRAND_TOTAL', {'sector': 'ALL'})
for model in ['xgb', 'lgb', 'ensemble']:
    ms = df[df['model'] == model]
    row[f'acc_{model}'] = round(ms['dir_accuracy'].mean(), 4)
    row[f'sharpe_{model}'] = round(ms['sharpe'].mean(), 4)
    row[f'pf_{model}'] = round(ms['profit_factor'].mean(), 4)
for w in df['window'].unique():
    ws = df[df['window'] == w]
    row[f'acc_{w}'] = round(ws['dir_accuracy'].mean(), 4)
    row[f'sharpe_{w}'] = round(ws['sharpe'].mean(), 4)
row['best_combo'] = ''
row['best_combo_acc'] = ''
row['best_combo_sharpe'] = ''
stock_rows.append(row)

stock_df = pd.DataFrame(stock_rows)
stock_df.to_csv(os.path.join(RESULTS_DIR, 'final_stock_performance.csv'), index=False)
print(f"Saved final_stock_performance.csv")

# ============================================================================
# 3. PER WINDOW PERFORMANCE (across all stocks, all models)
# ============================================================================
window_rows = []
for w in ['70/20/10', '65/25/10', '60/25/15', '55/30/15', '50/30/20']:
    sub = df[df['window'] == w]
    row = agg_row(sub, 'window', w)
    # Per-model breakdown
    for model in ['xgb', 'lgb', 'ensemble']:
        ms = sub[sub['model'] == model]
        row[f'acc_{model}'] = round(ms['dir_accuracy'].mean(), 4)
        row[f'sharpe_{model}'] = round(ms['sharpe'].mean(), 4)
        row[f'pf_{model}'] = round(ms['profit_factor'].mean(), 4)
        row[f'pct_acc50_{model}'] = round((ms['dir_accuracy'] > 0.50).mean() * 100, 1)
    # Per-sector breakdown
    for sector_name, symbols in [('Banking', ['HDFCBANK','ICICIBANK','KOTAKBANK','SBIN','AXISBANK']),
                                  ('IT', ['TCS','INFY','HCLTECH','WIPRO','TECHM'])]:
        ss = sub[sub['symbol'].isin(symbols)]
        row[f'acc_{sector_name}'] = round(ss['dir_accuracy'].mean(), 4)
        row[f'sharpe_{sector_name}'] = round(ss['sharpe'].mean(), 4)
    window_rows.append(row)

# Grand total
row = agg_row(df, 'window', 'ALL_WINDOWS_TOTAL')
for model in ['xgb', 'lgb', 'ensemble']:
    ms = df[df['model'] == model]
    row[f'acc_{model}'] = round(ms['dir_accuracy'].mean(), 4)
    row[f'sharpe_{model}'] = round(ms['sharpe'].mean(), 4)
    row[f'pf_{model}'] = round(ms['profit_factor'].mean(), 4)
    row[f'pct_acc50_{model}'] = round((ms['dir_accuracy'] > 0.50).mean() * 100, 1)
for sector_name, symbols in [('Banking', ['HDFCBANK','ICICIBANK','KOTAKBANK','SBIN','AXISBANK']),
                              ('IT', ['TCS','INFY','HCLTECH','WIPRO','TECHM'])]:
    ss = df[df['symbol'].isin(symbols)]
    row[f'acc_{sector_name}'] = round(ss['dir_accuracy'].mean(), 4)
    row[f'sharpe_{sector_name}'] = round(ss['sharpe'].mean(), 4)
window_rows.append(row)

window_df = pd.DataFrame(window_rows)
window_df.to_csv(os.path.join(RESULTS_DIR, 'final_window_performance.csv'), index=False)
print(f"Saved final_window_performance.csv")

# ============================================================================
# 4. EVERY SINGLE ROW with totals appended (the master sheet)
# ============================================================================
# Start with the raw 150 rows
master = df.copy()
master['sector'] = master['symbol'].apply(
    lambda s: 'Banking' if s in ['HDFCBANK','ICICIBANK','KOTAKBANK','SBIN','AXISBANK'] else 'IT'
)
master['above_50_acc'] = (master['dir_accuracy'] > 0.50).astype(str)
master['positive_sharpe'] = (master['sharpe'] > 0).astype(str)
master['profitable'] = (master['profit_factor'] > 1.0).astype(str)

# Add summary rows
summary_configs = [
    # Per stock totals
    *[(df[df['symbol'] == s], f'{s}_AVG') for s in df['symbol'].unique()],
    # Per model totals
    *[(df[df['model'] == m], f'MODEL_{m.upper()}_AVG') for m in ['xgb', 'lgb', 'ensemble']],
    # Per window totals
    *[(df[df['window'] == w], f'WINDOW_{w}_AVG') for w in df['window'].unique()],
    # Sector totals
    (banking, 'SECTOR_BANKING_AVG'),
    (it, 'SECTOR_IT_AVG'),
    # Grand total
    (df, 'GRAND_TOTAL_AVG'),
]

summary_rows = []
for subset, label in summary_configs:
    row = {
        'symbol': label,
        'window': 'ALL' if 'MODEL' in label or 'SECTOR' in label or 'GRAND' in label else (
            label.replace('WINDOW_', '').replace('_AVG', '') if 'WINDOW' in label else 'ALL'
        ),
        'model': 'ALL' if 'WINDOW' in label or 'SECTOR' in label or 'GRAND' in label or label.endswith('_AVG') and not label.startswith('MODEL') else (
            label.replace('MODEL_', '').replace('_AVG', '').lower() if 'MODEL' in label else 'ALL'
        ),
        'sector': '',
    }
    for m in METRICS:
        row[m] = round(subset[m].mean(), 4)
    row['n_train'] = ''
    row['n_test'] = ''
    row['n_val'] = int(subset['n_val'].mean())
    row['n_features'] = 240
    row['ensemble_weight_xgb'] = ''
    row['ensemble_weight_lgb'] = ''
    row['above_50_acc'] = f"{(subset['dir_accuracy'] > 0.50).mean()*100:.1f}%"
    row['positive_sharpe'] = f"{(subset['sharpe'] > 0).mean()*100:.1f}%"
    row['profitable'] = f"{(subset['profit_factor'] > 1.0).mean()*100:.1f}%"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
master_full = pd.concat([master, summary_df], ignore_index=True)
master_full.to_csv(os.path.join(RESULTS_DIR, 'final_master_with_totals.csv'), index=False)
print(f"Saved final_master_with_totals.csv ({len(master_full)} rows: {len(master)} detail + {len(summary_df)} summary)")

# ============================================================================
# CONSOLE PRINTOUT
# ============================================================================
print("\n" + "=" * 100)
print("FINAL MODEL PERFORMANCE")
print("=" * 100)
cols = ['model', 'dir_accuracy_mean', 'dir_accuracy_std', 'sharpe_mean', 'profit_factor_mean',
        'win_rate_long_mean', 'win_rate_short_mean', 'rmse_mean', 'mae_mean',
        'pct_acc_above_50', 'pct_sharpe_positive', 'pct_profitable', 'n_runs']
print(model_df[cols].to_string(index=False))

print("\n" + "=" * 100)
print("FINAL STOCK PERFORMANCE")
print("=" * 100)
cols = ['symbol', 'sector', 'dir_accuracy_mean', 'dir_accuracy_std', 'sharpe_mean', 'profit_factor_mean',
        'acc_xgb', 'acc_lgb', 'acc_ensemble', 'sharpe_xgb', 'sharpe_lgb', 'sharpe_ensemble',
        'pct_acc_above_50', 'pct_sharpe_positive', 'pct_profitable', 'n_runs']
print(stock_df[cols].to_string(index=False))

print("\n" + "=" * 100)
print("FINAL WINDOW PERFORMANCE")
print("=" * 100)
cols = ['window', 'dir_accuracy_mean', 'dir_accuracy_std', 'sharpe_mean', 'profit_factor_mean',
        'acc_xgb', 'acc_lgb', 'acc_ensemble', 'acc_Banking', 'acc_IT',
        'sharpe_Banking', 'sharpe_IT',
        'pct_acc_above_50', 'pct_sharpe_positive', 'pct_profitable', 'n_runs']
print(window_df[cols].to_string(index=False))

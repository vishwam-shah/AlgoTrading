import pandas as pd
import json, os, glob, statistics
from collections import Counter

run_dir = 'V3/06_results/runs/20260307_141956'
all_wins = pd.read_csv(run_dir + '/all_windows_detail.csv')
model_cols = ['lgbm_acc','xgb_acc','lstm_acc','bilstm_acc','gru_acc',
              'cnn_lstm_acc','cnn_gru_acc','tcn_gru_acc','tcn_transformer_acc','nbeats_acc']

print("=" * 65)
print("BEST SINGLE-WINDOW ACCURACY PER MODEL (best any stock, any window)")
print("=" * 65)
for col in model_cols:
    row = all_wins.loc[all_wins[col].idxmax()]
    print(f"  {col:<25} {row[col]:.4f}  {row['symbol']}  win{int(row['window_id'])}  oos={row['oos_accuracy']:.4f}")

above60 = all_wins[all_wins[model_cols].gt(0.60).any(axis=1)]
print(f'\nWindows with any model >60%: {len(above60)}')
if len(above60) > 0:
    print(above60[['symbol','window_id','oos_accuracy'] + model_cols].to_string())

print('\n' + "=" * 65)
print("OOS ACCURACY DISTRIBUTION (per window, ensemble)")
print("=" * 65)
print(all_wins['oos_accuracy'].describe().round(4))

# Which windows are already >60% ensemble?
ens60 = all_wins[all_wins['oos_accuracy'] > 0.60]
print(f'\nEnsemble OOS > 60% count: {len(ens60)} / {len(all_wins)} windows')
if len(ens60) > 0:
    print(ens60[['symbol','window_id','train_ratio','train_size','test_size','oos_accuracy'] + model_cols[:4]].to_string())

print('\n' + "=" * 65)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 65)
# tp/(tp+fn) = recall = fraction of actual UP days predicted UP
# Use precision (dir_acc_up) and recall separately
rows = []
for f in glob.glob(run_dir + '/*/summary_row.json'):
    rows.append(json.load(open(f)))

ups = [r['avg_dir_acc_up'] for r in rows if 'avg_dir_acc_up' in r]
dns = [r['avg_dir_acc_down'] for r in rows if 'avg_dir_acc_down' in r]
print(f"avg precision-UP  (when model says UP, % correct):   {statistics.mean(ups):.4f}")
print(f"avg precision-DOWN (when model says DOWN, % correct): {statistics.mean(dns):.4f}")
print(f"DOWN precision < 0.50 (model hurts on DOWN): {sum(1 for d in dns if d < 0.5)} / {len(dns)} stocks")
print("→ Models systematically LOSE on DOWN predictions (upward market bias 2019-2026)")

# Precision/Recall from window detail
if 'tp' in all_wins.columns:
    all_wins['n_actual_up'] = all_wins['tp'] + all_wins['fn']
    all_wins['n_actual_dn'] = all_wins['tn'] + all_wins['fp']
    all_wins['up_fraction'] = all_wins['n_actual_up'] / (all_wins['n_actual_up'] + all_wins['n_actual_dn'])
    print(f"\nAverage fraction of UP days in test set: {all_wins['up_fraction'].mean():.4f}")
    print(f"(50% = balanced; >50% = bull-market bias in dataset)")

print('\n' + "=" * 65)
print("TRAINING DATA SIZE ANALYSIS")
print("=" * 65)
print(all_wins[['train_size','val_size','test_size']].describe().round(0))
print(f"\nDL training sequences (≈ train_size - 19 for seq_len=20):")
dl_train = all_wins['train_size'] - 19
print(f"  min DL train seqs: {dl_train.min():.0f}  max: {dl_train.max():.0f}  mean: {dl_train.mean():.0f}")

print('\n' + "=" * 65)
print("EARLY STOPPING: EPOCH CEILING PROBLEM")
print("=" * 65)
print("DL_ES_PATIENCE=7  DL_MAX_EPOCHS=50  DL_BATCH_SIZE=64")
print(f"  At mean train_size={all_wins['train_size'].mean():.0f} seqs, batch=64:")
steps_per_epoch = all_wins['train_size'].mean() / 64
print(f"  Steps/epoch: {steps_per_epoch:.1f}")
print(f"  Total steps if stops at patience 7 (worst case epoch 8): {8 * steps_per_epoch:.0f}")
print(f"  For financial data, models need 50-80+ epochs to converge")
print(f"  → Models stop at ~{7*steps_per_epoch:.0f}-{15*steps_per_epoch:.0f} steps — UNDERTRAINED")

print('\n' + "=" * 65)
print("N-BEATS INPUT DIMENSION PROBLEM")  
print("=" * 65)
print("SEQ_LEN=20, n_features=50 → input_size = 1000 dims flattened")
print("FC hidden = 256 → compression ratio 3.9x (can't reconstruct 1000-dim backcast)")
print("With n_features=20: input_size=400 → compression 1.56x (workable)")
print("With PCA to 20 PCs: input_size=400 → similar to original NBEATS paper")

print('\n' + "=" * 65)
print("WHAT MODEL TO BET ON FOR 60%")
print("=" * 65)
# Stocks where XGB or LGBM already exceeded 55%
for col in ['xgb_acc','lgbm_acc','lstm_acc']:
    high = all_wins[all_wins[col] > 0.60][['symbol','window_id',col,'train_size']]
    print(f"{col} windows > 60%: {len(high)}")
    if len(high) > 0: print(high.to_string())

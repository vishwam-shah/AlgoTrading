"""Quick analysis of pipeline results."""
import json, numpy as np
from pathlib import Path
from collections import Counter

base = Path(r'C:\Users\Home\Documents\AI_IN_STOCK_V2\V3\06_results\runs\20260301_000647')
rows = []
for d in sorted(base.iterdir()):
    sr = d / 'summary_row.json'
    if d.is_dir() and sr.exists():
        with open(sr) as f:
            r = json.load(f)
        if r.get('status') == 'ok':
            rows.append(r)

accs = [r['oos_accuracy'] for r in rows]
print(f'=== {len(rows)} stocks completed ===')
print(f'Ensemble OOS:  mean={np.mean(accs):.2%}  median={np.median(accs):.2%}  max={np.max(accs):.2%}  min={np.min(accs):.2%}')
print(f'Above 55%: {sum(1 for a in accs if a>=0.55)}  |  Above 52%: {sum(1 for a in accs if a>=0.52)}  |  Above 50%: {sum(1 for a in accs if a>=0.50)}  |  Below 50%: {sum(1 for a in accs if a<0.50)}')

models = {
    'avg_lgbm_acc': 'LightGBM', 'avg_xgb_acc': 'XGBoost',
    'avg_lstm_acc': 'LSTM', 'avg_bilstm_acc': 'BiLSTM', 'avg_gru_acc': 'GRU',
    'avg_cnn_lstm_acc': 'CNN_LSTM', 'avg_cnn_gru_acc': 'CNN_GRU'
}
print(f'\n=== Per-Model Avg Accuracy (across stocks) ===')
for col, name in models.items():
    vals = [r.get(col, 0) for r in rows if r.get(col, 0) > 0]
    if vals:
        print(f'  {name:10s}: mean={np.mean(vals):.2%}  median={np.median(vals):.2%}  max={np.max(vals):.2%}  min={np.min(vals):.2%}')

best_counts = Counter(r.get('best_model', '?') for r in rows)
print(f'\n=== Best Model Distribution ===')
for m, c in best_counts.most_common():
    print(f'  {m}: {c} stocks')

sorted_rows = sorted(rows, key=lambda r: r['oos_accuracy'], reverse=True)
print(f'\n=== Top 10 Stocks ===')
for r in sorted_rows[:10]:
    sym = r['symbol']
    acc = r['oos_accuracy']
    bm  = r.get('best_model', '?')
    ba  = r.get('best_model_acc', 0)
    print(f'  {sym:14s} ens={acc:.2%}  best={bm}({ba:.2%})')

print(f'\n=== Bottom 10 Stocks ===')
for r in sorted_rows[-10:]:
    sym = r['symbol']
    acc = r['oos_accuracy']
    bm  = r.get('best_model', '?')
    ba  = r.get('best_model_acc', 0)
    print(f'  {sym:14s} ens={acc:.2%}  best={bm}({ba:.2%})')

# Window-level analysis: does accuracy degrade over time?
print(f'\n=== Window-Level Analysis (sample: top stock) ===')
top = sorted_rows[0]['symbol']
wr = base / top / 'window_results.csv'
if wr.exists():
    import pandas as pd
    wdf = pd.read_csv(wr)
    for _, w in wdf.iterrows():
        print(f"  Win {int(w['window_id'])} ({w['train_ratio']:.0%}): "
              f"ens={w['oos_accuracy']:.2%}  lgb={w.get('lgbm_acc',0):.2%}  xgb={w.get('xgb_acc',0):.2%}  "
              f"lstm={w.get('lstm_acc',0):.2%}  bilstm={w.get('bilstm_acc',0):.2%}  gru={w.get('gru_acc',0):.2%}")

# Are all models just around 50%?
print(f'\n=== Key Insight: Per-Window Model Accuracy Distribution ===')
all_window_accs = []
for d in base.iterdir():
    wr = d / 'window_results.csv'
    if wr.exists():
        import pandas as pd
        wdf = pd.read_csv(wr)
        for _, w in wdf.iterrows():
            for c in ['lgbm_acc','xgb_acc','lstm_acc','bilstm_acc','gru_acc','cnn_lstm_acc','cnn_gru_acc']:
                v = w.get(c, 0)
                if v > 0:
                    all_window_accs.append(v)

all_window_accs = np.array(all_window_accs)
print(f'  Total model-window observations: {len(all_window_accs)}')
print(f'  Mean: {np.mean(all_window_accs):.2%}  Median: {np.median(all_window_accs):.2%}')
print(f'  Above 60%: {np.mean(all_window_accs>0.60):.1%}')
print(f'  Above 55%: {np.mean(all_window_accs>0.55):.1%}')
print(f'  Above 50%: {np.mean(all_window_accs>0.50):.1%}')
print(f'  Below 45%: {np.mean(all_window_accs<0.45):.1%}')

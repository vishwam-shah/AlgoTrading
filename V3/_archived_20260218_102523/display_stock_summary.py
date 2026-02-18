"""
Display Stock-Wise Summary in Clean Format
"""

import pandas as pd

# Load stock summary
df = pd.read_csv(r'C:\Users\Home\Documents\AI_IN_STOCK_V2\V3\results\stock_wise_summary.csv')

print("="*140)
print(" " * 50 + "STOCK-WISE PERFORMANCE SUMMARY")
print(" " * 40 + "(Averaged Across All 7 Window Configs & 3 Models)")
print("="*140)

print("\n📊 DETAILED METRICS PER STOCK:\n")

# Format and display
for idx, row in df.iterrows():
    print(f"\n{'─'*140}")
    print(f"🏢 {row['symbol']:12s}")
    print(f"{'─'*140}")
    print(f"  ✓ Directional Accuracy  : {row['dir_accuracy']:.4f} ({row['dir_accuracy']*100:.2f}%)")
    print(f"  ✓ Win Rate Long         : {row['win_rate_long']:.4f} ({row['win_rate_long']*100:.2f}%)")
    print(f"  ✓ Win Rate Short        : {row['win_rate_short']:.4f} ({row['win_rate_short']*100:.2f}%)")
    print(f"  ✓ Win Rate Combined     : {row['win_rate_combined']:.4f} ({row['win_rate_combined']*100:.2f}%)")
    print(f"  ✓ Sharpe Ratio          : {row['sharpe']:.4f}")
    print(f"  ✓ Profit Factor         : {row['profit_factor']:.4f}")
    print(f"  ✓ RMSE                  : {row['rmse']:.6f}")
    print(f"  ✓ MAE                   : {row['mae']:.6f}")

# Overall averages
print(f"\n{'═'*140}")
print(f" " * 55 + "PORTFOLIO OVERALL AVERAGES")
print(f"{'═'*140}\n")

avg_dir_acc = df['dir_accuracy'].mean()
avg_win_long = df['win_rate_long'].mean()
avg_win_short = df['win_rate_short'].mean()
avg_win_combined = df['win_rate_combined'].mean()
avg_sharpe = df['sharpe'].mean()
avg_pf = df['profit_factor'].mean()
avg_rmse = df['rmse'].mean()
avg_mae = df['mae'].mean()

print(f"  📈 Average Directional Accuracy  : {avg_dir_acc:.4f} ({avg_dir_acc*100:.2f}%)")
print(f"  📈 Average Win Rate Long         : {avg_win_long:.4f} ({avg_win_long*100:.2f}%)")
print(f"  📈 Average Win Rate Short        : {avg_win_short:.4f} ({avg_win_short*100:.2f}%)")
print(f"  📈 Average Win Rate Combined     : {avg_win_combined:.4f} ({avg_win_combined*100:.2f}%)")
print(f"  📈 Average Sharpe Ratio          : {avg_sharpe:.4f}")
print(f"  📈 Average Profit Factor         : {avg_pf:.4f}")
print(f"  📈 Average RMSE                  : {avg_rmse:.6f}")
print(f"  📈 Average MAE                   : {avg_mae:.6f}")

# Rankings
print(f"\n{'═'*140}")
print(f" " * 60 + "TOP 5 RANKINGS")
print(f"{'═'*140}")

print(f"\n🏆 TOP 5 BY DIRECTIONAL ACCURACY:")
top_acc = df.nlargest(5, 'dir_accuracy')[['symbol', 'dir_accuracy']]
for i, row in enumerate(top_acc.itertuples(), 1):
    print(f"  {i}. {row.symbol:12s}: {row.dir_accuracy:.4f} ({row.dir_accuracy*100:.2f}%)")

print(f"\n🏆 TOP 5 BY COMBINED WIN RATE:")
top_wr = df.nlargest(5, 'win_rate_combined')[['symbol', 'win_rate_combined']]
for i, row in enumerate(top_wr.itertuples(), 1):
    print(f"  {i}. {row.symbol:12s}: {row.win_rate_combined:.4f} ({row.win_rate_combined*100:.2f}%)")

print(f"\n🏆 TOP 5 BY PROFIT FACTOR:")
top_pf = df.nlargest(5, 'profit_factor')[['symbol', 'profit_factor']]
for i, row in enumerate(top_pf.itertuples(), 1):
    pf_val = row.profit_factor
    gain_per_loss = f"₹{pf_val:.2f} gained per ₹1 lost"
    print(f"  {i}. {row.symbol:12s}: {pf_val:.4f} ({gain_per_loss})")

print(f"\n🏆 TOP 5 BY SHARPE RATIO:")
top_sharpe = df.nlargest(5, 'sharpe')[['symbol', 'sharpe']]
for i, row in enumerate(top_sharpe.itertuples(), 1):
    print(f"  {i}. {row.symbol:12s}: {row.sharpe:.4f}")

print(f"\n{'═'*140}")
print("📖 PROFIT FACTOR EXPLANATION:")
print(f"{'═'*140}")
print("""
Profit Factor (PF) = Total Gains / Total Losses

• PF > 1.0  → Profitable (you make more than you lose)
• PF = 1.0  → Break-even
• PF < 1.0  → Losing money

Examples:
• PF = 1.50 → For every ₹100 you lose, you make ₹150 (50% net profit)
• PF = 2.00 → For every ₹100 you lose, you make ₹200 (100% net profit) - EXCELLENT!
• PF = 0.80 → For every ₹100 you lose, you only make ₹80 (20% net loss) - BAD

Current Portfolio Average: {:.4f} (For every ₹1 lost, we make ₹{:.2f})
""".format(avg_pf, avg_pf))

print(f"{'═'*140}")
print("✅ SUMMARY COMPLETE - Full data saved to: V3/results/stock_wise_summary.csv")
print(f"{'═'*140}")

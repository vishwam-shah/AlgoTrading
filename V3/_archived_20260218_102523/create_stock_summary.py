"""
Create Stock-Wise Summary with Combined Win Rates and All Metric Averages
"""

import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv(r'C:\Users\Home\Documents\AI_IN_STOCK_V2\V3\results\all_stocks_summary.csv')

print("=" * 120)
print("STOCK-WISE PERFORMANCE SUMMARY")
print("=" * 120)

# Create stock-wise aggregation
stock_summary = df.groupby('symbol').agg({
    'dir_accuracy': 'mean',
    'win_rate_long': 'mean',
    'win_rate_short': 'mean',
    'sharpe': 'mean',
    'profit_factor': 'mean',
    'rmse': 'mean',
    'mae': 'mean',
    'n_features': 'first',  # Same for all rows
}).reset_index()

# Calculate combined win rate (average of long and short)
stock_summary['win_rate_combined'] = (stock_summary['win_rate_long'] + stock_summary['win_rate_short']) / 2

# Reorder columns
stock_summary = stock_summary[[
    'symbol',
    'dir_accuracy',
    'win_rate_long',
    'win_rate_short',
    'win_rate_combined',
    'sharpe',
    'profit_factor',
    'rmse',
    'mae',
    'n_features'
]]

# Round for display
stock_summary_display = stock_summary.copy()
for col in ['dir_accuracy', 'win_rate_long', 'win_rate_short', 'win_rate_combined', 'sharpe', 'profit_factor', 'rmse', 'mae']:
    stock_summary_display[col] = stock_summary_display[col].round(4)

print("\n📊 AVERAGE METRICS PER STOCK (Across All Windows & Models)")
print("-" * 120)
print(stock_summary_display.to_string(index=False))

# Overall portfolio average
print("\n" + "=" * 120)
print("📈 OVERALL PORTFOLIO AVERAGES")
print("=" * 120)
overall = {
    'Directional Accuracy': f"{stock_summary['dir_accuracy'].mean():.4f} ({stock_summary['dir_accuracy'].mean()*100:.2f}%)",
    'Win Rate Long': f"{stock_summary['win_rate_long'].mean():.4f} ({stock_summary['win_rate_long'].mean()*100:.2f}%)",
    'Win Rate Short': f"{stock_summary['win_rate_short'].mean():.4f} ({stock_summary['win_rate_short'].mean()*100:.2f}%)",
    'Win Rate Combined': f"{stock_summary['win_rate_combined'].mean():.4f} ({stock_summary['win_rate_combined'].mean()*100:.2f}%)",
    'Sharpe Ratio': f"{stock_summary['sharpe'].mean():.4f}",
    'Profit Factor': f"{stock_summary['profit_factor'].mean():.4f}",
    'RMSE': f"{stock_summary['rmse'].mean():.6f}",
    'MAE': f"{stock_summary['mae'].mean():.6f}",
}

for metric, value in overall.items():
    print(f"{metric:25s}: {value}")

# Best performing stocks
print("\n" + "=" * 120)
print("🏆 TOP 5 STOCKS BY METRIC")
print("=" * 120)

metrics_to_rank = {
    'Directional Accuracy': ('dir_accuracy', False),
    'Combined Win Rate': ('win_rate_combined', False),
    'Sharpe Ratio': ('sharpe', False),
    'Profit Factor': ('profit_factor', False),
    'Lowest RMSE': ('rmse', True),  # Lower is better
}

for metric_name, (col, ascending) in metrics_to_rank.items():
    top5 = stock_summary.nlargest(5, col) if not ascending else stock_summary.nsmallest(5, col)
    print(f"\n{metric_name}:")
    for i, row in enumerate(top5.itertuples(), 1):
        value = getattr(row, col)
        print(f"  {i}. {row.symbol:12s}: {value:.4f}")

# Save to CSV
output_file = r'C:\Users\Home\Documents\AI_IN_STOCK_V2\V3\results\stock_wise_summary.csv'
stock_summary.to_csv(output_file, index=False)
print(f"\n💾 Saved to: {output_file}")

# Explanation section
print("\n" + "=" * 120)
print("📖 METRIC DEFINITIONS & INTERPRETATION")
print("=" * 120)

explanations = """
1. **Directional Accuracy**: 
   - Percentage of times the model correctly predicted the direction (up/down) of next-day returns
   - Range: 0.0 to 1.0 (higher is better)
   - >0.50 is better than random guessing
   
2. **Win Rate Long**: 
   - Success rate when model predicted LONG (price will go up)
   - Percentage of profitable long trades
   
3. **Win Rate Short**: 
   - Success rate when model predicted SHORT (price will go down)
   - Percentage of profitable short trades
   
4. **Win Rate Combined**: 
   - Average of long and short win rates
   - Overall profitability across both directions
   
5. **Sharpe Ratio**: 
   - Risk-adjusted returns (return per unit of risk)
   - >1.0 is good, >2.0 is excellent
   - Negative means losses
   
6. **Profit Factor (PF)**: 
   - HOW IT WORKS: PF = (Sum of Profits) / (Sum of Losses)
   - Interpretation:
     * PF > 1.0: Profitable strategy (gains > losses)
     * PF = 1.0: Break-even
     * PF < 1.0: Losing strategy (losses > gains)
     * PF = 1.5: For every ₹1 lost, you make ₹1.50
     * PF = 2.0: For every ₹1 lost, you make ₹2.00 (excellent)
   - Example: If you made ₹1000 in winning trades and lost ₹500 in losing trades:
     PF = 1000 / 500 = 2.0
   
7. **RMSE (Root Mean Squared Error)**: 
   - Average magnitude of prediction errors
   - Lower is better (more accurate predictions)
   
8. **MAE (Mean Absolute Error)**: 
   - Average absolute difference between predictions and actual values
   - Lower is better
"""

print(explanations)

print("\n" + "=" * 120)
print("✅ ANALYSIS COMPLETE")
print("=" * 120)

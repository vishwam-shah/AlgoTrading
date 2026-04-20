'use client';
import { useEffect, useState } from 'react';
import { TrendingUp, Target } from 'lucide-react';
import { cn } from '@/lib/utils';
import { apiFetch } from '@/hooks/useData';

interface BacktestStock {
  symbol: string; oos_accuracy: number; tradeable: boolean;
  n_trades: number; total_return: number; win_rate: number;
  avg_win_pct: number; avg_loss_pct: number;
  profit_factor: number; sharpe: number; max_drawdown: number;
  up_signal_acc: number;
}

export interface BacktestResults {
  n_profitable: number; n_tradeable: number;
  portfolio_sharpe: number; portfolio_win_rate: number;
  portfolio_total_ret: number; portfolio_max_dd: number;
  portfolio_curve: { date: string; equity: number; daily_return: number }[];
  stocks: BacktestStock[];
  cost_model: string;
  bootstrap_significant?: boolean;
  bootstrap_ci_lower?: number;
  bootstrap_ci_upper?: number;
  bootstrap_acc_mean?: number;
  bootstrap_n_signals?: number;
  nifty_return?: number | null;
  nifty_start_date?: string;
  nifty_end_date?: string;
}

interface Props {
  runId?: string;
  onLoaded?: (data: BacktestResults | null) => void;
}

function MetricCard({ label, value, sub, color = '' }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className="text-[10px] text-muted-foreground uppercase tracking-wide">{label}</div>
      <div className={cn('text-xl font-bold font-mono mt-1 tabular-nums', color)}>{value}</div>
      {sub && <div className="text-xs text-muted-foreground mt-0.5">{sub}</div>}
    </div>
  );
}

export default function BacktestPanel({ runId, onLoaded }: Props) {
  const [backtest, setBacktest] = useState<BacktestResults | null>(null);

  useEffect(() => {
    const url = runId ? `/api/v3/runs/${runId}/backtest` : '/api/v3/runs/latest/backtest';
    apiFetch<BacktestResults>(url).then(d => {
      const result = d?.stocks ? d : null;
      setBacktest(result);
      onLoaded?.(result);
    });
  }, [runId]);

  if (!backtest) return (
    <div className="rounded-xl border border-border bg-card p-8 flex flex-col items-center gap-2 text-muted-foreground">
      <Target className="h-7 w-7 opacity-30" />
      <p className="text-sm">No backtest data yet</p>
      <p className="text-xs opacity-60">Run the pipeline first — backtest runs automatically</p>
    </div>
  );

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <Target className="h-4 w-4 text-green-400" /> Simulated Trade Backtest
          <span className="text-[10px] text-muted-foreground font-normal ml-1">{backtest.cost_model}</span>
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-8 gap-3">
          <MetricCard label="Tradeable Stocks" value={`${backtest.n_tradeable}`}
            sub={`of ${backtest.n_profitable} Sharpe>0`}
            color={backtest.n_tradeable >= 5 ? 'text-green-400' : 'text-yellow-400'} />
          <MetricCard label="Portfolio Sharpe" value={backtest.portfolio_sharpe.toFixed(2)}
            sub="avg of Sharpe>0 stocks"
            color={backtest.portfolio_sharpe >= 1.0 ? 'text-green-400' : backtest.portfolio_sharpe >= 0.5 ? 'text-yellow-400' : 'text-red-400'} />
          <MetricCard label="Avg Win Rate" value={`${(backtest.portfolio_win_rate * 100).toFixed(1)}%`}
            sub="profitable stocks only"
            color={backtest.portfolio_win_rate >= 0.55 ? 'text-green-400' : ''} />
          <MetricCard label="Portfolio Return" value={`${(backtest.portfolio_total_ret * 100).toFixed(1)}%`}
            sub="avg across profitable"
            color={backtest.portfolio_total_ret > 0 ? 'text-green-400' : 'text-red-400'} />
          <MetricCard label="NIFTY50 B&H" value={backtest.nifty_return != null ? `${(backtest.nifty_return * 100).toFixed(1)}%` : '—'}
            sub={backtest.nifty_start_date ? `${backtest.nifty_start_date.slice(0,10)} → ${backtest.nifty_end_date?.slice(0,10)}` : 'same period'}
            color={backtest.nifty_return != null ? (backtest.portfolio_total_ret > (backtest.nifty_return ?? 0) ? 'text-green-400' : 'text-yellow-400') : ''} />
          <MetricCard label="Bootstrap CI" value={backtest.bootstrap_acc_mean ? `${(backtest.bootstrap_acc_mean * 100).toFixed(1)}%` : '—'}
            sub={backtest.bootstrap_ci_lower != null ? `95% CI [${(backtest.bootstrap_ci_lower * 100).toFixed(1)}%, ${(backtest.bootstrap_ci_upper! * 100).toFixed(1)}%]` : 'pending'}
            color={backtest.bootstrap_significant ? 'text-green-400' : 'text-yellow-400'} />
          <MetricCard label="Avg Max Drawdown" value={`${(backtest.portfolio_max_dd * 100).toFixed(1)}%`}
            sub="profitable stocks"
            color={backtest.portfolio_max_dd < 0.15 ? 'text-green-400' : backtest.portfolio_max_dd < 0.25 ? 'text-yellow-400' : 'text-red-400'} />
          <MetricCard label="Round-trip Cost" value="0.25%" sub="STT+brokerage+slippage" />
        </div>
      </div>

      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-green-400" />
            <span className="text-sm font-semibold">Per-Stock Simulated P&L</span>
          </div>
          <span className="text-[10px] text-muted-foreground">sorted by Sharpe · conf ≥52%</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="bg-muted/10 text-muted-foreground">
              <tr>{['Symbol','OOS Acc','Trades','Total Ret','Win Rate','Avg Win','Avg Loss','Sharpe','MaxDD','UP Acc'].map(h =>
                <th key={h} className="text-left px-3 py-2 font-medium whitespace-nowrap">{h}</th>
              )}</tr>
            </thead>
            <tbody className="divide-y divide-border/30">
              {[...backtest.stocks]
                .filter(s => s.tradeable)
                .sort((a, b) => b.sharpe - a.sharpe)
                .slice(0, 20)
                .map(s => (
                  <tr key={s.symbol} className={cn('hover:bg-muted/10',
                    s.sharpe > 0.5 ? 'bg-green-500/5' : s.sharpe < 0 ? 'bg-red-500/5' : ''
                  )}>
                    <td className="px-3 py-2 font-mono font-bold">{s.symbol}</td>
                    <td className="px-3 py-2 font-mono">{(s.oos_accuracy * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 font-mono text-muted-foreground">{s.n_trades}</td>
                    <td className={cn('px-3 py-2 font-mono font-bold', s.total_return > 0 ? 'text-green-400' : 'text-red-400')}>
                      {s.total_return > 0 ? '+' : ''}{(s.total_return * 100).toFixed(1)}%
                    </td>
                    <td className="px-3 py-2 font-mono">{(s.win_rate * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 font-mono text-green-400">+{s.avg_win_pct.toFixed(2)}%</td>
                    <td className="px-3 py-2 font-mono text-red-400">{s.avg_loss_pct.toFixed(2)}%</td>
                    <td className={cn('px-3 py-2 font-mono font-bold',
                      s.sharpe >= 1.0 ? 'text-green-400' : s.sharpe >= 0.5 ? 'text-yellow-400' : s.sharpe > 0 ? 'text-muted-foreground' : 'text-red-400'
                    )}>{s.sharpe.toFixed(2)}</td>
                    <td className={cn('px-3 py-2 font-mono', s.max_drawdown > 0.30 ? 'text-red-400' : '')}>{(s.max_drawdown * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 font-mono text-muted-foreground">{(s.up_signal_acc * 100).toFixed(1)}%</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

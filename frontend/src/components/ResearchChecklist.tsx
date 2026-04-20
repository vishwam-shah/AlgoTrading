'use client';
import { FlaskConical } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SummaryStock {
  symbol: string;
  oos_accuracy: number;
  binary_dir_acc?: number;
  sharpe?: number;
  n_trades?: number;
  tradeable?: boolean;
}
interface BacktestResults {
  n_profitable: number;
  bootstrap_significant?: boolean;
  bootstrap_ci_lower?: number;
  bootstrap_ci_upper?: number;
  bootstrap_acc_mean?: number;
  nifty_return?: number | null;
}

interface Props {
  stocks: SummaryStock[];
  backtest: BacktestResults | null;
  paperSession: any;
}

export default function ResearchChecklist({ stocks, backtest, paperSession }: Props) {
  const nTrained   = stocks.length;
  const avgAcc     = nTrained > 0 ? stocks.reduce((s, r) => s + (r.oos_accuracy ?? 0), 0) / nTrained : 0;
  const above54    = stocks.filter(s => (s.oos_accuracy ?? 0) >= 0.54).length;
  const above52    = stocks.filter(s => (s.oos_accuracy ?? 0) >= 0.52).length;

  // Binary direction accuracy (up/down correct regardless of 5-class)
  const hasBinaryAcc  = stocks.some(s => s.binary_dir_acc != null);
  const avgBinaryAcc  = hasBinaryAcc
    ? stocks.reduce((s, r) => s + (r.binary_dir_acc ?? 0), 0) / nTrained
    : 0;

  // Backtest present if stocks have sharpe ratios computed
  const hasBacktest = stocks.some(s => s.sharpe != null && (s.sharpe ?? 0) > 0);

  const items = [
    { label: 'Walk-forward OOS validation',                                              done: true },
    { label: `${nTrained} stocks trained across sectors`,                                done: nTrained > 0 },
    { label: `Avg accuracy > 50% (${(avgAcc * 100).toFixed(1)}%)`,                      done: avgAcc >= 0.50 },
    { label: `Avg accuracy > 52% — publication bar (${(avgAcc * 100).toFixed(1)}%)`,    done: avgAcc >= 0.52 },
    { label: `Avg accuracy > 54% — trading edge (${(avgAcc * 100).toFixed(1)}%)`,       done: avgAcc >= 0.54 },
    { label: `${above54}/${nTrained} stocks above 54% OOS`,                             done: above54 >= 10 },
    { label: 'Backtest P&L / Sharpe ratio tracked',                                     done: hasBacktest || !!backtest },
    { label: `Binary direction accuracy measured (${(avgBinaryAcc * 100).toFixed(1)}%)`, done: hasBinaryAcc },
    { label: `Transaction cost model (${hasBacktest || backtest ? '0.25% RT' : 'pending'})`, done: hasBacktest || !!backtest },
    { label: `Statistical significance (bootstrap CI${backtest?.bootstrap_ci_lower != null ? ` ≥${((backtest.bootstrap_ci_lower ?? 0) * 100).toFixed(1)}%` : ''})`,
      done: !!backtest?.bootstrap_significant },
    { label: 'Paper trading started',                                                    done: !!paperSession },
    { label: `NIFTY buy-and-hold benchmark comparison${backtest?.nifty_return != null ? ` (NIFTY ${((backtest.nifty_return ?? 0) * 100).toFixed(1)}%)` : ''}`,
      done: backtest?.nifty_return != null },
  ];

  const doneCount = items.filter(i => i.done).length;

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center gap-2">
        <FlaskConical className="h-4 w-4 text-purple-400" />
        <span className="text-sm font-semibold">Research Readiness Checklist</span>
        <span className="ml-auto text-xs text-muted-foreground font-mono">{doneCount}/{items.length}</span>
        <div className="w-20 bg-muted/30 rounded-full h-1.5">
          <div className="h-1.5 rounded-full bg-purple-500 transition-all"
            style={{ width: `${(doneCount / items.length) * 100}%` }} />
        </div>
      </div>
      <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
        {items.map(item => (
          <div key={item.label} className={cn('flex items-center gap-2 px-3 py-2 rounded-lg',
            item.done ? 'bg-green-500/10' : 'bg-muted/20'
          )}>
            <span className={item.done ? 'text-green-400' : 'text-muted-foreground/50'}>
              {item.done ? '✓' : '○'}
            </span>
            <span className={item.done ? 'text-foreground' : 'text-muted-foreground'}>{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

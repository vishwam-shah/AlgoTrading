'use client';
import { useEffect, useState } from 'react';
import { Clock, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { apiFetch } from '@/hooks/useData';

interface ModeSummary {
  portfolio_total_return?: number;
  portfolio_sharpe?: number;
  portfolio_max_dd?: number;
  avg_per_stock_return?: number;
  bootstrap_acc_mean?: number;
  bootstrap_ci_lower?: number;
  bootstrap_ci_upper?: number;
  bootstrap_n_signals?: number;
  nifty_return?: number;
  error?: string;
}
interface CompareResp {
  run_id: string;
  modes: Record<string, ModeSummary>;
}

const TIMING_LABEL: Record<string, string> = {
  same_close:  'Same-close (legacy)',
  next_close:  'Next-close',
  next_open:   'Next-open (T+1, default)',
};

export default function BacktestTimingToggle({ runId }: { runId: string | null }) {
  const [data, setData] = useState<CompareResp | null>(null);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    if (!runId) return;
    setLoading(true);
    const d = await apiFetch<CompareResp>(
      `/api/v3/backtest/timing-compare/${runId}`,
      120_000
    );
    if (d) setData(d);
    setLoading(false);
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  const modes = data?.modes ?? {};

  return (
    <div className="rounded-xl border border-border bg-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold uppercase tracking-wide">
            Backtest Entry-Timing Comparison
          </h3>
        </div>
        <button
          onClick={load}
          disabled={!runId || loading}
          className="text-xs px-2 py-1 rounded border border-border hover:bg-muted disabled:opacity-50 flex items-center gap-1"
        >
          {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
          {loading ? 'Computing…' : 'Recompute'}
        </button>
      </div>

      <p className="text-xs text-muted-foreground">
        Compares how the same predictions perform under each execution
        assumption. <span className="text-amber-400">same_close</span> is the
        legacy unrealistic path (entry on signal day);{' '}
        <span className="text-green-400">next_open</span> is realistic T+1 fill
        and matches the live runner.
      </p>

      {!runId ? (
        <div className="text-sm text-muted-foreground py-4 text-center">
          Pick a run from the run-selector first.
        </div>
      ) : (
        <div className="overflow-auto rounded border border-border">
          <table className="w-full text-xs">
            <thead className="bg-muted/40">
              <tr className="text-left">
                <th className="px-2 py-1.5">Timing</th>
                <th className="px-2 py-1.5 text-right">Total Return</th>
                <th className="px-2 py-1.5 text-right">Sharpe</th>
                <th className="px-2 py-1.5 text-right">Max DD</th>
                <th className="px-2 py-1.5 text-right">Avg/stock</th>
                <th className="px-2 py-1.5 text-right">Bootstrap Acc</th>
                <th className="px-2 py-1.5 text-right">95% CI</th>
                <th className="px-2 py-1.5 text-right">N signals</th>
              </tr>
            </thead>
            <tbody>
              {(['same_close', 'next_close', 'next_open'] as const).map((mode) => {
                const m = modes[mode] ?? {};
                if (m.error) {
                  return (
                    <tr key={mode} className="border-t border-border/50">
                      <td className="px-2 py-1.5 font-mono">{TIMING_LABEL[mode]}</td>
                      <td colSpan={7} className="px-2 py-1.5 text-red-400 text-xs">
                        error: {m.error}
                      </td>
                    </tr>
                  );
                }
                return (
                  <tr key={mode} className={cn('border-t border-border/50',
                    mode === 'next_open' && 'bg-green-500/5')}>
                    <td className="px-2 py-1.5 font-mono">{TIMING_LABEL[mode]}</td>
                    <td className={cn('px-2 py-1.5 text-right tabular-nums',
                      (m.portfolio_total_return ?? 0) > 0 ? 'text-green-400' : 'text-red-400')}>
                      {pct(m.portfolio_total_return)}
                    </td>
                    <td className={cn('px-2 py-1.5 text-right tabular-nums',
                      (m.portfolio_sharpe ?? 0) > 1 ? 'text-green-400'
                      : (m.portfolio_sharpe ?? 0) < 0 ? 'text-red-400' : 'text-amber-400')}>
                      {fmt(m.portfolio_sharpe, 3)}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-amber-400">
                      {pct(m.portfolio_max_dd)}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums">
                      {pct(m.avg_per_stock_return)}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums">
                      {pct(m.bootstrap_acc_mean)}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-muted-foreground">
                      [{pct(m.bootstrap_ci_lower)}, {pct(m.bootstrap_ci_upper)}]
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums">
                      {m.bootstrap_n_signals ?? '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {modes.next_open && modes.same_close && !modes.next_open.error && !modes.same_close.error && (
        <div className="rounded border border-border bg-muted/20 p-3 text-xs">
          <div className="font-semibold text-muted-foreground uppercase tracking-wide mb-1">
            Headline
          </div>
          T+1 vs legacy:{' '}
          <span className="font-mono">
            ΔSharpe ={' '}
            <span className={cn(
              ((modes.next_open.portfolio_sharpe ?? 0) - (modes.same_close.portfolio_sharpe ?? 0)) > 0
                ? 'text-green-400' : 'text-red-400'
            )}>
              {(((modes.next_open.portfolio_sharpe ?? 0) -
                (modes.same_close.portfolio_sharpe ?? 0))).toFixed(3)}
            </span>
          </span>
          {' · '}
          <span className="font-mono">
            ΔMax DD ={' '}
            {(((modes.next_open.portfolio_max_dd ?? 0) -
              (modes.same_close.portfolio_max_dd ?? 0)) * 100).toFixed(2)}pp
          </span>
        </div>
      )}
    </div>
  );
}

function fmt(n: number | undefined, dp = 2) {
  if (n === undefined || n === null || Number.isNaN(n)) return '—';
  return n.toFixed(dp);
}
function pct(n: number | undefined) {
  if (n === undefined || n === null || Number.isNaN(n)) return '—';
  return `${(n * 100).toFixed(2)}%`;
}

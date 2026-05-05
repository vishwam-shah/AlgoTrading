'use client';
import { useEffect, useState } from 'react';
import { FlaskConical, RefreshCw, Play } from 'lucide-react';
import { cn } from '@/lib/utils';
import { apiFetch } from '@/hooks/useData';

interface RobustnessResp {
  run_id: string;
  tables: {
    cost_slippage?: any[];
    turnover?: any[];
    hold_horizon?: any[];
    regime?: any[];
    calibration_drift?: any[];
  };
  summary?: any;
}

export default function RobustnessTab({ runId }: { runId: string | null }) {
  const [data, setData] = useState<RobustnessResp | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);

  const load = async () => {
    if (!runId) return;
    setLoading(true);
    const d = await apiFetch<RobustnessResp>(`/api/v3/robustness/${runId}`);
    if (d) setData(d);
    setLoading(false);
  };

  const triggerRun = async () => {
    if (!runId) return;
    setRunning(true);
    await fetch(`/api/v3/robustness/${runId}/run`, { method: 'POST' });
    setTimeout(() => {
      setRunning(false);
      load();
    }, 5000);
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  const cost = data?.tables.cost_slippage ?? [];
  const turn = data?.tables.turnover ?? [];
  const horiz = data?.tables.hold_horizon ?? [];
  const regime = data?.tables.regime ?? [];

  // Best turnover row by sharpe_mean (top 10)
  const turnTop = [...turn]
    .filter((r) => r.sharpe_mean !== undefined)
    .sort((a, b) => (b.sharpe_mean ?? 0) - (a.sharpe_mean ?? 0))
    .slice(0, 10);

  return (
    <div className="rounded-xl border border-border bg-card p-5 space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold uppercase tracking-wide">
            Robustness Suite (exp9)
          </h3>
          {runId && (
            <span className="text-xs text-muted-foreground font-mono">{runId}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={triggerRun}
            disabled={!runId || running}
            className="text-xs px-2 py-1 rounded border border-border hover:bg-muted disabled:opacity-50 flex items-center gap-1"
          >
            <Play className="h-3 w-3" />
            {running ? 'Running…' : 'Re-run suite'}
          </button>
          <button
            onClick={load}
            className="text-xs p-1 rounded border border-border hover:bg-muted"
          >
            <RefreshCw className={cn('h-3 w-3', loading && 'animate-spin')} />
          </button>
        </div>
      </div>

      {!data?.tables || Object.keys(data.tables).length === 0 ? (
        <div className="text-sm text-muted-foreground py-6 text-center">
          No robustness results yet for this run. Hit{' '}
          <kbd className="px-1 py-0.5 border border-border rounded">Re-run suite</kbd> or
          execute{' '}
          <code className="font-mono text-xs">
            python V3/08_experiments/exp9_robustness_suite.py --run-id {runId}
          </code>
          .
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* A. Cost / slippage */}
          {cost.length > 0 && (
            <Section title="A. Cost / Slippage Sensitivity">
              <SimpleTable
                headers={['RT cost', '# stocks', 'Sharpe', 'Median', '% pos']}
                rows={cost.map((r) => [
                  `${(r.cost_round_trip * 100).toFixed(2)}%`,
                  r.n_stocks,
                  fmtNum(r.sharpe_mean, 'sharpe'),
                  fmtNum(r.sharpe_med),
                  pct(r.pct_pos),
                ])}
              />
            </Section>
          )}

          {/* C. Hold-horizon */}
          {horiz.length > 0 && (
            <Section title="C. Hold-Horizon Sensitivity">
              <SimpleTable
                headers={['Hold (d)', '# stocks', 'Trades', 'Sharpe', 'Median', 'Avg ret']}
                rows={horiz.map((r) => [
                  r.hold_days,
                  r.n_stocks,
                  r.n_trades_mean,
                  fmtNum(r.sharpe_mean, 'sharpe'),
                  fmtNum(r.sharpe_med),
                  pct(r.ret_mean),
                ])}
              />
            </Section>
          )}

          {/* B. Turnover top 10 */}
          {turnTop.length > 0 && (
            <Section title="B. Turnover (top-10 by Sharpe)">
              <SimpleTable
                headers={['min_conf', 'meta_thr', 'Trades', 'Sharpe', '% pos']}
                rows={turnTop.map((r) => [
                  r.min_conf,
                  r.meta_thr,
                  r.n_trades_mean,
                  fmtNum(r.sharpe_mean, 'sharpe'),
                  pct(r.pct_pos),
                ])}
              />
            </Section>
          )}

          {/* D. Regime */}
          {regime.length > 0 && (
            <Section title="D. Regime-Conditional Replay">
              <SimpleTable
                headers={['Regime', 'Trades', 'Sharpe', 'Win rate', 'Avg ret']}
                rows={regime.map((r) => [
                  r.regime,
                  r.n_trades,
                  fmtNum(r.sharpe, 'sharpe'),
                  pct(r.win_rate),
                  pct(r.avg_ret),
                ])}
              />
            </Section>
          )}
        </div>
      )}

      {data?.summary && (
        <div className="rounded border border-border bg-muted/20 p-3 text-xs">
          <div className="font-semibold text-muted-foreground uppercase tracking-wide mb-1">
            Headline
          </div>
          {data.summary.cost_slippage_breakeven_cost !== undefined && (
            <div>
              Break-even RT cost:{' '}
              <span className="font-mono">
                {(data.summary.cost_slippage_breakeven_cost * 100).toFixed(3)}%
              </span>
            </div>
          )}
          {data.summary.best_turnover && (
            <div>
              Best turnover: min_conf={data.summary.best_turnover.min_conf}, meta=
              {data.summary.best_turnover.meta_thr} → Sharpe{' '}
              <span className="font-mono">
                {data.summary.best_turnover.sharpe_mean}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
        {title}
      </div>
      {children}
    </div>
  );
}

function SimpleTable({
  headers,
  rows,
}: {
  headers: string[];
  rows: (string | number | null)[][];
}) {
  return (
    <div className="overflow-auto rounded border border-border">
      <table className="w-full text-xs">
        <thead className="bg-muted/40">
          <tr>
            {headers.map((h, i) => (
              <th
                key={h}
                className={cn('px-2 py-1', i === 0 ? 'text-left' : 'text-right')}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-border/50">
              {r.map((cell, j) => (
                <td
                  key={j}
                  className={cn(
                    'px-2 py-1 tabular-nums',
                    j === 0 ? 'text-left' : 'text-right'
                  )}
                >
                  {cell ?? '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function fmtNum(n: number | null | undefined, mode?: 'sharpe') {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  if (mode === 'sharpe') {
    const cls = n > 1 ? 'text-green-400' : n < 0 ? 'text-red-400' : 'text-amber-400';
    return <span className={cls}>{n.toFixed(3)}</span>;
  }
  return n.toFixed(3);
}

function pct(n: number | null | undefined) {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return `${(n * 100).toFixed(1)}%`;
}

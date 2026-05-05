'use client';
import { Shield, ShieldCheck, ShieldAlert, RefreshCw, Check, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { usePolling } from '@/hooks/useData';

interface Check {
  name: string;
  passed: boolean;
  value: number | string;
  target: number | string;
  detail?: string;
}
interface Promotion {
  evaluated_at: string;
  decision: 'go' | 'no-go';
  checks: Check[];
  summary: Record<string, number | null>;
  thresholds: Record<string, number>;
}

const LABEL: Record<string, string> = {
  min_paper_trades:   'Closed paper trades',
  min_paper_days:     'Days since first trade',
  min_rolling_sharpe: 'Rolling 30-trade Sharpe',
  max_rolling_dd:     'Rolling 30-trade max DD',
  max_slip_bps:       'Avg slippage drift (bps)',
  min_fill_rate:      'Fill rate',
  max_brier_drift:    'Calibration drift',
};

export default function PromotionGatePanel() {
  const { data, refresh, loading } = usePolling<Promotion>('/api/v3/promotion/status', 60_000);

  const decision = data?.decision ?? 'no-go';
  const isGo = decision === 'go';
  const checks = data?.checks ?? [];
  const passed = checks.filter((c) => c.passed).length;

  return (
    <div className="rounded-xl border border-border bg-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isGo ? (
            <ShieldCheck className="h-4 w-4 text-green-400" />
          ) : (
            <ShieldAlert className="h-4 w-4 text-amber-400" />
          )}
          <h3 className="text-sm font-semibold uppercase tracking-wide">
            Paper → Live Promotion Gate
          </h3>
        </div>
        <button
          onClick={refresh}
          className="text-xs p-1 rounded border border-border hover:bg-muted"
          aria-label="refresh"
        >
          <RefreshCw className={cn('h-3 w-3', loading && 'animate-spin')} />
        </button>
      </div>

      <div
        className={cn(
          'rounded-lg border p-3 flex items-center justify-between',
          isGo ? 'bg-green-500/10 border-green-500/40' : 'bg-amber-500/10 border-amber-500/40'
        )}
      >
        <div>
          <div className={cn('text-2xl font-bold', isGo ? 'text-green-400' : 'text-amber-400')}>
            {decision.toUpperCase()}
          </div>
          <div className="text-xs text-muted-foreground">
            {passed}/{checks.length} checks passed · last evaluated{' '}
            {data?.evaluated_at ? new Date(data.evaluated_at).toLocaleString() : '—'}
          </div>
        </div>
        <Shield className={cn('h-12 w-12', isGo ? 'text-green-400/30' : 'text-amber-400/30')} />
      </div>

      <div className="space-y-1">
        {checks.map((c) => (
          <div
            key={c.name}
            className="flex items-center justify-between text-sm border-b border-border/50 py-1.5"
          >
            <div className="flex items-center gap-2">
              {c.passed ? (
                <Check className="h-3.5 w-3.5 text-green-400" />
              ) : (
                <X className="h-3.5 w-3.5 text-red-400" />
              )}
              <span>{LABEL[c.name] ?? c.name}</span>
            </div>
            <div className="flex items-center gap-3 font-mono text-xs">
              <span className={cn(c.passed ? 'text-green-400' : 'text-red-400')}>
                {c.value}
              </span>
              <span className="text-muted-foreground">target {c.target}</span>
            </div>
          </div>
        ))}
      </div>

      {data?.summary && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 text-xs pt-2 border-t border-border">
          {Object.entries(data.summary).map(([k, v]) => (
            <div key={k} className="rounded border border-border p-2">
              <div className="text-muted-foreground uppercase tracking-wide text-[10px]">{k}</div>
              <div className="font-mono tabular-nums">{v ?? '—'}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

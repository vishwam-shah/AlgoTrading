'use client';
import { useEffect, useState } from 'react';
import { Activity, Play, CheckCircle2, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { apiFetch } from '@/hooks/useData';
import { toast } from 'sonner';

interface PaperSession {
  session_id: string;
  note?: string;
  qualified_stocks?: { symbol: string; sharpe?: number }[];
  todays_signals?: any[];
  initial_cash?: number;
}

interface Props {
  onLoaded?: (session: PaperSession | null) => void;
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

export default function PaperTradingPanel({ onLoaded }: Props) {
  const [status, setStatus]   = useState<'idle' | 'starting' | 'started'>('idle');
  const [session, setSession] = useState<PaperSession | null>(null);

  useEffect(() => {
    apiFetch<PaperSession>('/api/v3/paper/latest')
      .then(d => {
        const s = d?.session_id ? d : null;
        setSession(s);
        if (s) setStatus('started');
        onLoaded?.(s);
      })
      .catch(() => {});
  }, []);

  async function startSession() {
    setStatus('starting');
    try {
      const res = await fetch('/api/v3/paper/start', { method: 'POST' });
      const r = res.ok ? await res.json() : null;
      if (r?.session_id) {
        setSession(r);
        setStatus('started');
        onLoaded?.(r);
        toast.success(`Paper trading started — ${r.qualified_stocks?.length ?? 0} stocks`);
      } else {
        setStatus('idle');
        toast.error('Failed to start session');
      }
    } catch (e: any) {
      toast.error(`Failed: ${e.message}`);
      setStatus('idle');
    }
  }

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-indigo-400" />
          <span className="text-sm font-semibold">Paper Trading</span>
          <span className="text-[10px] text-muted-foreground">top stocks by Sharpe ≥ 0.5</span>
        </div>
        <button
          onClick={startSession}
          disabled={status === 'starting'}
          className={cn(
            'flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium transition-all',
            status === 'started'
              ? 'bg-green-500/10 text-green-400 border border-green-500/20'
              : 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 hover:bg-indigo-500/20'
          )}>
          {status === 'starting' ? (
            <><Activity className="h-3 w-3 animate-pulse" /> Starting…</>
          ) : status === 'started' ? (
            <><CheckCircle2 className="h-3 w-3" /> Session Active</>
          ) : (
            <><Play className="h-3 w-3" /> Start Paper Trading</>
          )}
        </button>
      </div>

      {session ? (
        <div className="p-4 space-y-3">
          {session.note && <div className="text-xs text-muted-foreground">{session.note}</div>}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard label="Session"           value={session.session_id?.slice(-6) ?? '—'} sub="active" />
            <MetricCard label="Stocks"            value={`${session.qualified_stocks?.length ?? 0}`} sub="Sharpe≥0.5" />
            <MetricCard label="Today UP signals"  value={`${(session.todays_signals ?? []).length}`}
              sub="to monitor" color={(session.todays_signals ?? []).length > 0 ? 'text-green-400' : ''} />
            <MetricCard label="Capital"           value={`₹${((session.initial_cash ?? 0) / 100000).toFixed(1)}L`} sub="paper" />
          </div>
          {(session.qualified_stocks ?? []).length > 0 && (
            <div className="flex flex-wrap gap-1.5 pt-1">
              {session.qualified_stocks!.map(s => (
                <span key={s.symbol} className="px-2 py-1 rounded-lg bg-green-500/10 text-green-400 border border-green-500/20 text-xs font-mono">
                  {s.symbol} <span className="text-green-300/60">S:{s.sharpe?.toFixed(2)}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="p-6 flex flex-col items-center gap-2 text-muted-foreground">
          <AlertCircle className="h-6 w-6 opacity-30" />
          <p className="text-sm">No paper trading session yet</p>
          <p className="text-xs opacity-60">Requires pipeline run with backtest results</p>
        </div>
      )}
    </div>
  );
}

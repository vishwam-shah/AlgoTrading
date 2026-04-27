'use client';
import { useEffect, useState } from 'react';
import { Activity, FileClock, TrendingUp, TrendingDown, Beaker, Radio, RefreshCw, Trash2, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { apiFetch } from '@/hooks/useData';
import { toast } from 'sonner';

interface Fill {
  symbol: string;
  side: 'BUY' | 'SELL';
  ordered_qty: number;
  filled_qty: number;
  avg_price: number;
  order_value: number;
  brokerage: number;
  stt: number;
  other_charges: number;
  status: string;
  placed_at: string;
  filled_at?: string;
  order_id: string;
}
interface Position {
  symbol: string;
  net_qty: number;
  avg_price: number;
  invested: number;
  last_price: number;
  last_trade_at: string;
  is_paper: boolean;
  n_trades: number;
}
interface LatestSession {
  file: string;
  timestamp: string;
  mode: 'paper' | 'live';
  n_fills: number;
  n_total: number;
  total_value: number;
  total_charges: number;
  fills: Fill[];
}
interface Overview {
  mode: 'paper' | 'live';
  scope: 'latest' | 'all';
  latest_session: LatestSession | null;
  positions: Position[];
  totals: { fills: number; deployed: number; charges: number; sessions: number };
}

type Scope = 'latest' | 'all';

interface Props {
  livePrices?: Record<string, number>;
}

export default function ExecutionPanel({ livePrices = {} }: Props) {
  const [data,      setData]      = useState<Overview | null>(null);
  const [loading,   setLoading]   = useState(true);
  const [resetting, setResetting] = useState(false);
  const [scope,     setScope]     = useState<Scope>('latest');
  const [tab,       setTab]       = useState<'positions' | 'latest'>('positions');

  const load = async (s: Scope = scope) => {
    setLoading(true);
    const r = await apiFetch<Overview>(`/api/v3/execution/overview?scope=${s}`);
    if (r) setData(r);
    setLoading(false);
  };

  useEffect(() => { load(scope); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [scope]);

  const resetPortfolio = async (keepLatest: boolean) => {
    const msg = keepLatest
      ? 'Archive all sessions except the latest?'
      : 'Archive ALL execution sessions and reset paper portfolio to zero?';
    if (!window.confirm(msg + '\n\nFiles are moved to execution_logs/archive/ — nothing is deleted.')) return;
    setResetting(true);
    try {
      const res = await fetch(`/api/v3/execution/reset?keep_latest=${keepLatest}`, { method: 'POST' });
      const j   = await res.json();
      if (res.ok) {
        toast.success(`Archived ${j.archived} session${j.archived === 1 ? '' : 's'}${j.kept ? ` · kept ${j.kept}` : ''}`);
        await load(scope);
      } else {
        toast.error(j.detail || 'Reset failed');
      }
    } catch {
      toast.error('Backend unreachable');
    } finally {
      setResetting(false);
    }
  };

  if (loading && !data) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 flex items-center justify-center min-h-[200px] text-muted-foreground text-sm">
        <RefreshCw className="h-4 w-4 animate-spin mr-2" /> Loading execution data…
      </div>
    );
  }

  if (!data || (data.totals.fills === 0 && !data.latest_session)) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 flex flex-col items-center justify-center gap-3 text-center min-h-[200px]">
        <FileClock className="h-8 w-8 text-muted-foreground/40" />
        <p className="text-sm text-muted-foreground">No execution sessions yet</p>
        <p className="text-xs text-muted-foreground/60">
          Run <span className="font-mono">python V3/05_live_trading/daily_runner.py --mode morning</span> to simulate paper fills.
        </p>
      </div>
    );
  }

  const mode       = data.mode;
  const liveValue  = data.positions.reduce((s, p) => s + p.net_qty * (livePrices[p.symbol] || p.last_price || p.avg_price), 0);
  const invested   = data.positions.reduce((s, p) => s + p.invested, 0);
  const pnl        = liveValue - invested;
  const pnlPct     = invested > 0 ? (pnl / invested) * 100 : 0;

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      {/* Header: mode + totals + actions */}
      <div className="border-b border-border p-4 flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3 flex-wrap">
          <div className={cn(
            'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border',
            mode === 'live'
              ? 'bg-red-500/10 text-red-400 border-red-500/20'
              : 'bg-blue-500/10 text-blue-400 border-blue-500/20'
          )}>
            {mode === 'live'
              ? <><Radio  className="h-3 w-3" /> LIVE MODE</>
              : <><Beaker className="h-3 w-3" /> PAPER MODE</>}
          </div>
          <div className="text-xs text-muted-foreground">
            {data.totals.sessions} sessions · {data.totals.fills} fills
          </div>

          {/* Scope toggle */}
          <div className="flex items-center gap-0.5 bg-muted/40 rounded-lg p-0.5">
            {([
              { id: 'latest' as const, label: 'Latest Session' },
              { id: 'all'    as const, label: 'All-time'       },
            ]).map(s => (
              <button key={s.id} onClick={() => setScope(s.id)}
                className={cn(
                  'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all',
                  scope === s.id
                    ? 'bg-background text-foreground shadow-sm border border-border'
                    : 'text-muted-foreground hover:text-foreground'
                )}>
                {s.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button onClick={() => load(scope)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-border hover:bg-muted/50 text-xs transition-colors">
            <RefreshCw className={cn('h-3 w-3', loading && 'animate-spin')} />
            Refresh
          </button>
          <button onClick={() => resetPortfolio(true)}
            disabled={resetting || data.totals.sessions <= 1}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-border hover:bg-muted/50 text-xs transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            title="Archive all sessions except the latest one">
            {resetting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
            Archive Old
          </button>
          <button onClick={() => resetPortfolio(false)}
            disabled={resetting || data.totals.sessions === 0}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-red-500/30 bg-red-500/5 text-red-400 hover:bg-red-500/10 text-xs transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            title="Archive ALL sessions and reset paper portfolio to zero">
            {resetting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
            Reset All
          </button>
        </div>
      </div>

      {/* Summary grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-border border-b border-border">
        {[
          { label: 'Positions',      value: data.positions.length.toString(), color: '' },
          { label: 'Invested',       value: `₹${invested.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`, color: '' },
          { label: 'Live Value',     value: `₹${liveValue.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`, color: '' },
          { label: 'P&L (unreal.)',  value: `${pnl >= 0 ? '+' : ''}₹${Math.abs(pnl).toLocaleString('en-IN', { maximumFractionDigits: 0 })} (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%)`,
            color: pnl >= 0 ? 'text-green-400' : 'text-red-400' },
        ].map(item => (
          <div key={item.label} className="px-4 py-3">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wide">{item.label}</div>
            <div className={cn('text-base font-bold font-mono tabular-nums mt-0.5', item.color)}>{item.value}</div>
          </div>
        ))}
      </div>

      {/* Sub-tabs */}
      <div className="flex border-b border-border bg-muted/5">
        {([
          { id: 'positions', label: `Positions (${data.positions.length})` },
          { id: 'latest',    label: `Latest Session${data.latest_session ? ` · ${data.latest_session.n_fills} fills` : ''}` },
        ] as { id: 'positions' | 'latest'; label: string }[]).map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={cn(
              'px-4 py-2.5 text-xs font-medium border-b-2 transition-all',
              tab === t.id
                ? 'border-primary text-primary'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            )}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Positions table */}
      {tab === 'positions' && (
        data.positions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 text-muted-foreground gap-2">
            <Activity className="h-7 w-7 opacity-30" />
            <p className="text-sm">No open positions</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/10 border-b border-border">
                <tr>
                  {['Symbol','Net Qty','Avg Cost','LTP','Invested','Live Value','P&L','Trades'].map(h => (
                    <th key={h} className="text-left px-4 py-2.5 text-xs font-medium text-muted-foreground uppercase tracking-wide whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40">
                {data.positions.map(p => {
                  const live    = livePrices[p.symbol] || p.last_price || p.avg_price;
                  const curVal  = p.net_qty * live;
                  const rowPnl  = curVal - p.invested;
                  const rowPct  = p.invested > 0 ? (rowPnl / p.invested) * 100 : 0;
                  return (
                    <tr key={p.symbol} className="hover:bg-muted/10 transition-colors">
                      <td className="px-4 py-3 font-mono font-bold">
                        <div className="flex items-center gap-2">
                          {p.symbol}
                          {p.is_paper && (
                            <span className="px-1 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[9px] font-semibold">PAPER</span>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 font-mono tabular-nums">{p.net_qty}</td>
                      <td className="px-4 py-3 font-mono text-muted-foreground tabular-nums">₹{p.avg_price.toFixed(2)}</td>
                      <td className="px-4 py-3 font-mono tabular-nums">
                        {live > 0 ? `₹${live.toFixed(2)}` : '—'}
                      </td>
                      <td className="px-4 py-3 font-mono tabular-nums">₹{p.invested.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</td>
                      <td className="px-4 py-3 font-mono tabular-nums">₹{curVal.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</td>
                      <td className="px-4 py-3">
                        <div className={cn('flex items-center gap-1 font-mono tabular-nums text-xs font-bold',
                          rowPnl >= 0 ? 'text-green-400' : 'text-red-400')}>
                          {rowPnl >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                          {rowPnl >= 0 ? '+' : ''}₹{Math.abs(rowPnl).toFixed(0)}
                          <span className="text-[10px] opacity-70">({rowPct >= 0 ? '+' : ''}{rowPct.toFixed(2)}%)</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-muted-foreground text-xs">{p.n_trades}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )
      )}

      {/* Latest session fills */}
      {tab === 'latest' && data.latest_session && (
        <div>
          <div className="px-4 py-2.5 text-xs text-muted-foreground bg-muted/5 border-b border-border flex items-center justify-between gap-4">
            <span className="font-mono">{data.latest_session.file}</span>
            <span>
              ₹{data.latest_session.total_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })} deployed ·
              ₹{data.latest_session.total_charges.toFixed(2)} charges
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/10 border-b border-border">
                <tr>
                  {['Symbol','Side','Qty','Price','Value','Charges','Status','Time'].map(h => (
                    <th key={h} className="text-left px-4 py-2.5 text-xs font-medium text-muted-foreground uppercase tracking-wide whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40">
                {data.latest_session.fills.map(f => {
                  const chg = (f.brokerage || 0) + (f.stt || 0) + (f.other_charges || 0);
                  const ok  = f.status === 'FILLED';
                  return (
                    <tr key={f.order_id} className="hover:bg-muted/10 transition-colors">
                      <td className="px-4 py-2.5 font-mono font-bold">{f.symbol}</td>
                      <td className="px-4 py-2.5">
                        <span className={cn('px-1.5 py-0.5 rounded text-[10px] font-semibold',
                          f.side === 'BUY' ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'
                        )}>{f.side}</span>
                      </td>
                      <td className="px-4 py-2.5 font-mono tabular-nums">{f.filled_qty}/{f.ordered_qty}</td>
                      <td className="px-4 py-2.5 font-mono tabular-nums">₹{(f.avg_price || 0).toFixed(2)}</td>
                      <td className="px-4 py-2.5 font-mono tabular-nums">₹{(f.order_value || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}</td>
                      <td className="px-4 py-2.5 font-mono text-muted-foreground tabular-nums text-xs">₹{chg.toFixed(2)}</td>
                      <td className="px-4 py-2.5">
                        <span className={cn('px-1.5 py-0.5 rounded text-[10px] font-semibold',
                          ok ? 'bg-green-500/10 text-green-400' : 'bg-muted text-muted-foreground'
                        )}>{f.status}</span>
                      </td>
                      <td className="px-4 py-2.5 text-muted-foreground text-xs font-mono">
                        {(f.filled_at || f.placed_at || '').split('T')[1]?.slice(0,8) || '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Footer hint */}
      <div className="px-4 py-2.5 text-[11px] text-muted-foreground bg-muted/5 border-t border-border">
        {mode === 'paper'
          ? 'Paper mode: simulated fills — no real orders on Angel One. Flip TRADING_MODE=live in .env to go live.'
          : 'Live mode: orders are being placed on Angel One. Reconcile with holdings at 15:45 IST.'}
        {' · '}Positions scope: {scope === 'latest' ? 'latest session only' : `all ${data.totals.sessions} sessions`}
        {' · '}All-time charges: ₹{data.totals.charges.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
      </div>
    </div>
  );
}

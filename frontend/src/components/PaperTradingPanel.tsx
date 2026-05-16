'use client';
import { useEffect, useState, useCallback } from 'react';
import {
  Activity, Play, TrendingUp, TrendingDown, RefreshCw,
  DollarSign, BarChart2, Target, AlertCircle, ChevronDown, ChevronUp,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

// ── Types ──────────────────────────────────────────────────────────────────────

interface Summary {
  nav: number;
  cash: number;
  holdings_value: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_return_pct: number;
  starting_capital: number;
  win_rate_pct: number;
  sharpe: number;
  max_drawdown_pct: number;
  open_positions: number;
  closed_trades: number;
  as_of: string;
}

interface Position {
  symbol: string;
  qty: number;
  avg_entry: number;
  ltp: number;
  unrealized: number;
  unrealized_pct: number;
  change_pct: number;
  hold_days: number;
  entry_date: string;
}

interface ClosedTrade {
  symbol: string;
  qty: number;
  entry_price: number;
  exit_price: number;
  entry_date: string;
  exit_date: string;
  hold_days: number;
  net_pnl: number;
  ret_pct: number;
  win: boolean;
}

interface Signal {
  symbol: string;
  prob_up?: number;
  confidence?: number;
  direction?: string;
}

interface NavPoint { date: string; nav: number; }

interface Dashboard {
  summary: Summary;
  positions: Position[];
  closed_trades: ClosedTrade[];
  nav_curve: NavPoint[];
  todays_signals: Signal[];
}

// ── Helpers ────────────────────────────────────────────────────────────────────

const fmt  = (n: number) => n.toLocaleString('en-IN', { maximumFractionDigits: 2 });
const fmtL = (n: number) => `₹${(n / 100000).toFixed(2)}L`;
const pct  = (n: number) => `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
const pnlColor = (n: number) => n >= 0 ? 'text-green-400' : 'text-red-400';

function StatCard({ label, value, sub, color = '' }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="rounded-xl border border-border bg-card/60 p-3">
      <div className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">{label}</div>
      <div className={cn('text-lg font-bold font-mono tabular-nums', color)}>{value}</div>
      {sub && <div className="text-[10px] text-muted-foreground mt-0.5">{sub}</div>}
    </div>
  );
}

// ── Mini equity curve (SVG sparkline) ─────────────────────────────────────────

function EquityCurve({ data, startingCapital }: { data: NavPoint[]; startingCapital: number }) {
  if (data.length < 2) {
    return (
      <div className="h-24 flex items-center justify-center text-xs text-muted-foreground">
        No NAV history yet — positions needed
      </div>
    );
  }
  const W = 600; const H = 80;
  const navs = data.map(d => d.nav);
  const minN = Math.min(...navs, startingCapital * 0.97);
  const maxN = Math.max(...navs, startingCapital * 1.03);
  const xS = (i: number) => (i / (navs.length - 1)) * W;
  const yS = (v: number) => H - ((v - minN) / (maxN - minN || 1)) * H;

  // baseline (starting capital)
  const baseY = yS(startingCapital);
  const pts = navs.map((v, i) => `${xS(i)},${yS(v)}`).join(' ');
  const last = navs[navs.length - 1];
  const color = last >= startingCapital ? '#4ade80' : '#f87171';

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-24" preserveAspectRatio="none">
        {/* baseline */}
        <line x1="0" y1={baseY} x2={W} y2={baseY} stroke="#374151" strokeWidth="1" strokeDasharray="4,4" />
        {/* area fill */}
        <polygon
          points={`0,${H} ${pts} ${W},${H}`}
          fill={color} fillOpacity="0.08"
        />
        {/* line */}
        <polyline points={pts} fill="none" stroke={color} strokeWidth="2" />
      </svg>
      <div className="absolute top-1 left-2 text-[10px] text-muted-foreground">Equity curve</div>
      <div className="absolute top-1 right-2 text-[10px] font-mono" style={{ color }}>
        {fmtL(last)} ({pct((last - startingCapital) / startingCapital * 100)})
      </div>
    </div>
  );
}

// ── Positions table ────────────────────────────────────────────────────────────

function PositionsTable({ positions }: { positions: Position[] }) {
  if (positions.length === 0) {
    return (
      <div className="text-center py-6 text-xs text-muted-foreground">
        No open positions — signals appear here after morning buys
      </div>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted-foreground text-[10px] uppercase">
            <th className="text-left py-1.5 pr-3 font-medium">Symbol</th>
            <th className="text-right py-1.5 pr-3 font-medium">Qty</th>
            <th className="text-right py-1.5 pr-3 font-medium">Entry</th>
            <th className="text-right py-1.5 pr-3 font-medium">LTP</th>
            <th className="text-right py-1.5 pr-3 font-medium">Day%</th>
            <th className="text-right py-1.5 pr-3 font-medium">Unrealised</th>
            <th className="text-right py-1.5 pr-3 font-medium">P&amp;L%</th>
            <th className="text-right py-1.5 font-medium">Hold</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/40">
          {positions.map(p => (
            <tr key={p.symbol} className="hover:bg-muted/10">
              <td className="py-1.5 pr-3 font-mono font-semibold">{p.symbol}</td>
              <td className="py-1.5 pr-3 text-right">{p.qty}</td>
              <td className="py-1.5 pr-3 text-right font-mono">₹{fmt(p.avg_entry)}</td>
              <td className="py-1.5 pr-3 text-right font-mono font-semibold">₹{fmt(p.ltp)}</td>
              <td className={cn('py-1.5 pr-3 text-right', pnlColor(p.change_pct))}>
                {pct(p.change_pct)}
              </td>
              <td className={cn('py-1.5 pr-3 text-right font-mono', pnlColor(p.unrealized))}>
                ₹{fmt(p.unrealized)}
              </td>
              <td className={cn('py-1.5 pr-3 text-right font-semibold', pnlColor(p.unrealized_pct))}>
                {pct(p.unrealized_pct)}
              </td>
              <td className="py-1.5 text-right text-muted-foreground">{p.hold_days}d</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Closed trades ──────────────────────────────────────────────────────────────

function ClosedTradesTable({ trades }: { trades: ClosedTrade[] }) {
  const [expanded, setExpanded] = useState(false);
  const visible = expanded ? trades : trades.slice(0, 8);
  if (trades.length === 0) {
    return (
      <div className="text-center py-4 text-xs text-muted-foreground">
        No closed trades yet
      </div>
    );
  }
  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-muted-foreground text-[10px] uppercase">
              <th className="text-left py-1.5 pr-3 font-medium">Symbol</th>
              <th className="text-right py-1.5 pr-3 font-medium">Qty</th>
              <th className="text-right py-1.5 pr-3 font-medium">Entry</th>
              <th className="text-right py-1.5 pr-3 font-medium">Exit</th>
              <th className="text-right py-1.5 pr-3 font-medium">Return</th>
              <th className="text-right py-1.5 pr-3 font-medium">Net P&amp;L</th>
              <th className="text-right py-1.5 pr-3 font-medium">Hold</th>
              <th className="text-right py-1.5 font-medium">Exit date</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {visible.map((t, i) => (
              <tr key={i} className={cn('hover:bg-muted/10', t.win ? '' : 'opacity-75')}>
                <td className="py-1.5 pr-3">
                  <span className="font-mono font-semibold">{t.symbol}</span>
                  <span className={cn('ml-1.5 text-[9px] px-1 py-0.5 rounded',
                    t.win ? 'bg-green-500/15 text-green-400' : 'bg-red-500/15 text-red-400')}>
                    {t.win ? 'WIN' : 'LOSS'}
                  </span>
                </td>
                <td className="py-1.5 pr-3 text-right">{t.qty}</td>
                <td className="py-1.5 pr-3 text-right font-mono">₹{fmt(t.entry_price)}</td>
                <td className="py-1.5 pr-3 text-right font-mono">₹{fmt(t.exit_price)}</td>
                <td className={cn('py-1.5 pr-3 text-right font-semibold', pnlColor(t.ret_pct))}>
                  {pct(t.ret_pct)}
                </td>
                <td className={cn('py-1.5 pr-3 text-right font-mono', pnlColor(t.net_pnl))}>
                  ₹{fmt(t.net_pnl)}
                </td>
                <td className="py-1.5 pr-3 text-right text-muted-foreground">{t.hold_days}d</td>
                <td className="py-1.5 text-right text-muted-foreground">{t.exit_date?.slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {trades.length > 8 && (
        <button
          onClick={() => setExpanded(e => !e)}
          className="w-full mt-2 text-xs text-muted-foreground hover:text-foreground flex items-center justify-center gap-1 py-1">
          {expanded ? <><ChevronUp className="h-3 w-3" /> Show less</> : <><ChevronDown className="h-3 w-3" /> Show all {trades.length} trades</>}
        </button>
      )}
    </div>
  );
}

// ── Today's signals ────────────────────────────────────────────────────────────

function TodaysSignals({ signals }: { signals: Signal[] }) {
  if (signals.length === 0) {
    return <div className="text-xs text-muted-foreground text-center py-2">No UP signals today</div>;
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {signals.map(s => {
        const conf = s.prob_up ?? s.confidence ?? 0;
        return (
          <div key={s.symbol}
            className="px-2.5 py-1 rounded-lg bg-green-500/10 border border-green-500/25 text-xs font-mono">
            <span className="text-green-400 font-semibold">{s.symbol}</span>
            {conf > 0 && <span className="text-green-300/60 ml-1">{(conf * 100).toFixed(0)}%</span>}
          </div>
        );
      })}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

interface Props {
  onLoaded?: (session: any | null) => void;
}

export default function PaperTradingPanel({ onLoaded }: Props) {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [loading, setLoading]     = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState<'positions' | 'trades' | 'signals'>('positions');

  const load = useCallback(async (quiet = false) => {
    if (!quiet) setRefreshing(true);
    try {
      const res = await fetch('/api/v3/paper/dashboard');
      if (res.ok) {
        const d = await res.json();
        setDashboard(d);
        onLoaded?.(d);
      } else if (res.status === 404) {
        setDashboard(null);
      }
    } catch {
      // backend not reachable — keep current state
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [onLoaded]);

  useEffect(() => {
    load(true);
    // Auto-refresh every 60 seconds during market hours
    const timer = setInterval(() => load(true), 60_000);
    return () => clearInterval(timer);
  }, [load]);

  async function startSession() {
    try {
      const res = await fetch('/api/v3/paper/start', { method: 'POST' });
      if (res.ok) {
        toast.success('Paper trading session started');
        await load();
      } else {
        const err = await res.text();
        toast.error(`Failed: ${err}`);
      }
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  const s = dashboard?.summary;

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-indigo-400" />
          <span className="text-sm font-semibold">Paper Trading</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-500/15 text-yellow-400 border border-yellow-500/20">
            PAPER
          </span>
          {s && (
            <span className={cn('text-xs font-mono font-bold', pnlColor(s.total_return_pct))}>
              {pct(s.total_return_pct)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => load()}
            disabled={refreshing}
            className="p-1.5 rounded-lg hover:bg-muted/20 text-muted-foreground transition-colors">
            <RefreshCw className={cn('h-3.5 w-3.5', refreshing && 'animate-spin')} />
          </button>
          {!s && !loading && (
            <button
              onClick={startSession}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 hover:bg-indigo-500/20">
              <Play className="h-3 w-3" /> Start Session
            </button>
          )}
        </div>
      </div>

      {loading ? (
        <div className="p-8 flex items-center justify-center text-muted-foreground text-sm">
          <RefreshCw className="h-4 w-4 animate-spin mr-2" /> Loading dashboard…
        </div>
      ) : !s ? (
        <div className="p-8 flex flex-col items-center gap-2 text-muted-foreground">
          <AlertCircle className="h-6 w-6 opacity-30" />
          <p className="text-sm">No paper trading data yet</p>
          <p className="text-xs opacity-60">Run the pipeline first, then start a session</p>
          <button
            onClick={startSession}
            className="mt-2 flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 hover:bg-indigo-500/20">
            <Play className="h-3.5 w-3.5" /> Start Paper Trading
          </button>
        </div>
      ) : (
        <div className="p-4 space-y-4">

          {/* KPI cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
            <StatCard label="Portfolio NAV"    value={fmtL(s.nav)}
              sub={`of ${fmtL(s.starting_capital)}`} />
            <StatCard label="Cash"             value={fmtL(s.cash)} sub="available" />
            <StatCard label="Total Return"     value={pct(s.total_return_pct)}
              color={pnlColor(s.total_return_pct)} sub="vs starting capital" />
            <StatCard label="Realized P&L"     value={`₹${fmt(s.realized_pnl)}`}
              color={pnlColor(s.realized_pnl)} sub="closed trades" />
            <StatCard label="Unrealized P&L"   value={`₹${fmt(s.unrealized_pnl)}`}
              color={pnlColor(s.unrealized_pnl)} sub="open positions" />
            <StatCard label="Win Rate"         value={`${s.win_rate_pct.toFixed(1)}%`}
              color={s.win_rate_pct >= 50 ? 'text-green-400' : 'text-red-400'}
              sub={`${s.closed_trades} trades`} />
            <StatCard label="Sharpe"           value={s.sharpe.toFixed(2)}
              color={s.sharpe >= 1 ? 'text-green-400' : s.sharpe >= 0 ? 'text-yellow-400' : 'text-red-400'} />
            <StatCard label="Max Drawdown"     value={`-${s.max_drawdown_pct.toFixed(2)}%`}
              color={s.max_drawdown_pct > 10 ? 'text-red-400' : 'text-muted-foreground'} />
          </div>

          {/* Equity curve */}
          <div className="rounded-lg border border-border bg-card/40 p-3">
            <EquityCurve data={dashboard!.nav_curve} startingCapital={s.starting_capital} />
          </div>

          {/* Tabs */}
          <div className="flex gap-1 border-b border-border">
            {([
              { key: 'positions', label: `Open Positions (${s.open_positions})`,  icon: <BarChart2 className="h-3 w-3" /> },
              { key: 'trades',    label: `Closed Trades (${s.closed_trades})`,    icon: <Target     className="h-3 w-3" /> },
              { key: 'signals',   label: `Today's Signals (${dashboard!.todays_signals.length})`, icon: <TrendingUp className="h-3 w-3" /> },
            ] as const).map(tab => (
              <button key={tab.key} onClick={() => setActiveTab(tab.key)}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition-colors -mb-px',
                  activeTab === tab.key
                    ? 'border-indigo-400 text-indigo-400'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                )}>
                {tab.icon}{tab.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="min-h-[120px]">
            {activeTab === 'positions' && <PositionsTable positions={dashboard!.positions} />}
            {activeTab === 'trades'    && <ClosedTradesTable trades={dashboard!.closed_trades} />}
            {activeTab === 'signals'   && <TodaysSignals signals={dashboard!.todays_signals} />}
          </div>

          {/* Footer */}
          <div className="text-[10px] text-muted-foreground text-right">
            Live prices via NSE · Updated {new Date(s.as_of).toLocaleTimeString()} · 60s auto-refresh
          </div>
        </div>
      )}
    </div>
  );
}

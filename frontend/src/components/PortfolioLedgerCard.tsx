'use client';
import { Wallet, TrendingUp, TrendingDown, RefreshCw, FileLock2, Layers } from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';
import StatCard from './StatCard';
import { usePolling, apiFetch } from '@/hooks/useData';

interface OpenLot {
  symbol: string;
  qty: number;
  entry_price: number;
  entry_date: string;
  order_id: string;
}
interface ClosedTrade {
  symbol: string;
  qty: number;
  entry_price: number;
  exit_price: number;
  entry_date: string;
  exit_date: string;
  ret_pct: number;
  net_pnl: number;
  hold_days: number;
}
interface LedgerSnap {
  nav: number;
  cash: number;
  realized_pnl: number;
  unrealized_pnl: number;
  open_lots: OpenLot[] | number;
  closed_trades: number;
  open_symbols: string[];
  pending_orders: number;
  starting_capital: number;
  ret_total_pct: number;
  last_rebuilt_at: string;
  closed_trades_recent?: ClosedTrade[];
}

const fmtINR = (n: number) =>
  '₹' + (n ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 0 });

export default function PortfolioLedgerCard() {
  const { data, refresh, loading } = usePolling<LedgerSnap>('/api/v3/portfolio/state', 30_000);
  const [rebuilding, setRebuilding] = useState(false);

  const rebuild = async () => {
    setRebuilding(true);
    await apiFetch('/api/v3/portfolio/rebuild', 30_000);
    await refresh();
    setRebuilding(false);
  };

  const resetPaper = async () => {
    if (!window.confirm(
      'Archive all paper-trading history and start fresh?\n\n' +
      'This moves execution_log.parquet, all execution_*.json, all orders/*.json, ' +
      'and the ledger into execution_logs/archive/<timestamp>/. ' +
      'Nothing is deleted — you can restore by moving files back.\n\n' +
      'New starting capital: ₹5,00,000'
    )) return;
    setRebuilding(true);
    await fetch('/api/v3/execution/reset?full=true&starting_capital=500000', { method: 'POST' });
    await refresh();
    setRebuilding(false);
  };

  const openLots: OpenLot[] = Array.isArray(data?.open_lots)
    ? (data!.open_lots as OpenLot[])
    : [];
  const ret = data?.ret_total_pct ?? 0;
  const trend: 'up' | 'down' | 'neutral' = ret > 0 ? 'up' : ret < 0 ? 'down' : 'neutral';

  return (
    <div className="rounded-xl border border-border bg-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileLock2 className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold uppercase tracking-wide">
            Canonical Portfolio Ledger
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={rebuild}
            disabled={rebuilding}
            className="text-xs px-2 py-1 rounded border border-border hover:bg-muted disabled:opacity-50"
          >
            {rebuilding ? 'Rebuilding…' : 'Rebuild from log'}
          </button>
          <button
            onClick={resetPaper}
            disabled={rebuilding}
            className="text-xs px-2 py-1 rounded border border-amber-500/40 text-amber-400 hover:bg-amber-500/10 disabled:opacity-50"
            title="Archive all paper-trading state and start fresh"
          >
            Reset paper state
          </button>
          <button
            onClick={refresh}
            className="text-xs p-1 rounded border border-border hover:bg-muted"
            aria-label="refresh"
          >
            <RefreshCw className={cn('h-3 w-3', loading && 'animate-spin')} />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard label="NAV" value={fmtINR(data?.nav ?? 0)} icon={Wallet} highlight />
        <StatCard
          label="Total Return"
          value={`${ret > 0 ? '+' : ''}${ret.toFixed(2)}%`}
          trend={trend}
          icon={trend === 'up' ? TrendingUp : TrendingDown}
        />
        <StatCard label="Realized P&L" value={fmtINR(data?.realized_pnl ?? 0)} />
        <StatCard label="Cash" value={fmtINR(data?.cash ?? 0)} />
        <StatCard label="Open Lots" value={openLots.length} icon={Layers} />
        <StatCard label="Closed Trades" value={data?.closed_trades ?? 0} />
        <StatCard label="Pending Orders" value={data?.pending_orders ?? 0} />
        <StatCard
          label="Capital"
          value={fmtINR(data?.starting_capital ?? 0)}
          sub={data?.last_rebuilt_at ? `rebuilt ${new Date(data.last_rebuilt_at).toLocaleString()}` : ''}
        />
      </div>

      {openLots.length > 0 && (
        <div>
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
            Open Lots ({openLots.length})
          </div>
          <div className="overflow-auto max-h-72 rounded border border-border">
            <table className="w-full text-xs">
              <thead className="bg-muted/40 sticky top-0">
                <tr className="text-left">
                  <th className="px-2 py-1">Symbol</th>
                  <th className="px-2 py-1 text-right">Qty</th>
                  <th className="px-2 py-1 text-right">Entry ₹</th>
                  <th className="px-2 py-1">Entry Date</th>
                  <th className="px-2 py-1">Order</th>
                </tr>
              </thead>
              <tbody>
                {openLots.slice(0, 80).map((l, i) => (
                  <tr key={`${l.order_id}-${i}`} className="border-t border-border/50">
                    <td className="px-2 py-1 font-mono">{l.symbol}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{l.qty}</td>
                    <td className="px-2 py-1 text-right tabular-nums">
                      {l.entry_price.toFixed(2)}
                    </td>
                    <td className="px-2 py-1 text-muted-foreground">{l.entry_date}</td>
                    <td className="px-2 py-1 text-muted-foreground truncate max-w-[10rem]">
                      {l.order_id}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {(data?.closed_trades_recent ?? []).length > 0 && (
        <div>
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
            Recent Closed Trades
          </div>
          <div className="overflow-auto max-h-64 rounded border border-border">
            <table className="w-full text-xs">
              <thead className="bg-muted/40 sticky top-0">
                <tr className="text-left">
                  <th className="px-2 py-1">Symbol</th>
                  <th className="px-2 py-1 text-right">Qty</th>
                  <th className="px-2 py-1 text-right">Entry → Exit</th>
                  <th className="px-2 py-1 text-right">Return</th>
                  <th className="px-2 py-1 text-right">Net P&L</th>
                  <th className="px-2 py-1 text-right">Days</th>
                </tr>
              </thead>
              <tbody>
                {data!.closed_trades_recent!.map((t, i) => (
                  <tr key={i} className="border-t border-border/50">
                    <td className="px-2 py-1 font-mono">{t.symbol}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{t.qty}</td>
                    <td className="px-2 py-1 text-right tabular-nums">
                      {t.entry_price.toFixed(2)} → {t.exit_price.toFixed(2)}
                    </td>
                    <td
                      className={cn(
                        'px-2 py-1 text-right tabular-nums',
                        t.ret_pct > 0 ? 'text-green-400' : t.ret_pct < 0 ? 'text-red-400' : ''
                      )}
                    >
                      {t.ret_pct > 0 ? '+' : ''}
                      {t.ret_pct.toFixed(2)}%
                    </td>
                    <td className={cn('px-2 py-1 text-right tabular-nums', t.net_pnl > 0 ? 'text-green-400' : 'text-red-400')}>
                      {fmtINR(t.net_pnl)}
                    </td>
                    <td className="px-2 py-1 text-right tabular-nums">{t.hold_days}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

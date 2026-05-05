'use client';
import { LogOut, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { usePolling } from '@/hooks/useData';

interface ExitOrder {
  symbol: string;
  qty: number;
  price: number;
  reason: string;
  entry_date: string;
  entry_price: number;
  hold_days: number;
}
interface ExitsResp {
  exits: ExitOrder[];
  count: number;
  by_reason: Record<string, number>;
  evaluated_at: string;
}

const REASON_COLOR: Record<string, string> = {
  vol_stop:        'bg-red-500/15 text-red-400 border-red-500/40',
  trailing_stop:   'bg-amber-500/15 text-amber-400 border-amber-500/40',
  signal_decay:    'bg-purple-500/15 text-purple-400 border-purple-500/40',
  time_stop:       'bg-blue-500/15 text-blue-400 border-blue-500/40',
  partial_profit:  'bg-emerald-500/15 text-emerald-400 border-emerald-500/40',
  no_bars:         'bg-zinc-500/15 text-zinc-400 border-zinc-500/40',
  within_hold_window: 'bg-zinc-700 text-zinc-400 border-zinc-700',
};

export default function ExitsTodayTable() {
  const { data, refresh, loading } = usePolling<ExitsResp>('/api/v3/exits/today', 60_000);

  const exits = data?.exits ?? [];
  const byReason = data?.by_reason ?? {};

  return (
    <div className="rounded-xl border border-border bg-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <LogOut className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold uppercase tracking-wide">
            Exits Today (multi-rule policy)
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

      {Object.keys(byReason).length > 0 && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(byReason).map(([reason, n]) => (
            <span
              key={reason}
              className={cn(
                'text-xs px-2 py-1 rounded border font-mono',
                REASON_COLOR[reason] ?? 'bg-muted text-muted-foreground border-border'
              )}
            >
              {reason}: {n}
            </span>
          ))}
        </div>
      )}

      <div className="text-xs text-muted-foreground">
        {data?.evaluated_at && (
          <>Evaluated {new Date(data.evaluated_at).toLocaleString()} — </>
        )}
        {data?.count ?? 0} exit orders generated
      </div>

      {exits.length > 0 ? (
        <div className="overflow-auto max-h-[420px] rounded border border-border">
          <table className="w-full text-xs">
            <thead className="bg-muted/40 sticky top-0">
              <tr className="text-left">
                <th className="px-2 py-1">Symbol</th>
                <th className="px-2 py-1 text-right">Qty</th>
                <th className="px-2 py-1 text-right">Sell @</th>
                <th className="px-2 py-1 text-right">Entry @</th>
                <th className="px-2 py-1 text-right">P&L %</th>
                <th className="px-2 py-1">Reason</th>
                <th className="px-2 py-1">Entry Date</th>
              </tr>
            </thead>
            <tbody>
              {exits.map((e, i) => {
                const pnl = ((e.price - e.entry_price) / e.entry_price) * 100;
                return (
                  <tr key={i} className="border-t border-border/50">
                    <td className="px-2 py-1 font-mono">{e.symbol}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{e.qty}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{e.price.toFixed(2)}</td>
                    <td className="px-2 py-1 text-right tabular-nums">
                      {e.entry_price?.toFixed(2)}
                    </td>
                    <td className={cn('px-2 py-1 text-right tabular-nums',
                      pnl > 0 ? 'text-green-400' : pnl < 0 ? 'text-red-400' : '')}>
                      {pnl > 0 ? '+' : ''}{pnl.toFixed(2)}%
                    </td>
                    <td className="px-2 py-1">
                      <span className={cn(
                        'text-[10px] px-1.5 py-0.5 rounded border font-mono',
                        REASON_COLOR[e.reason] ?? 'bg-muted text-muted-foreground border-border'
                      )}>
                        {e.reason}
                      </span>
                    </td>
                    <td className="px-2 py-1 text-muted-foreground">{e.entry_date}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-sm text-muted-foreground py-4 text-center">
          No exit orders due. Either no open positions or none meet exit criteria.
        </div>
      )}
    </div>
  );
}

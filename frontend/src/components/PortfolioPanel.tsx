'use client';
import { useState } from 'react';
import { TrendingUp, TrendingDown, Wallet, RefreshCw, Send, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

interface Holding { symbol: string; qty: number; avg_price: number; ltp: number; pnl: number; }
interface Holdings { holdings: Holding[]; total_pnl: number; count: number; }
interface Funds    { funds: { available: number; net: number; used_margin: number } }

interface PortfolioPanelProps {
  holdings: Holdings | null;
  funds: Funds | null;
  prices: Record<string, number>;
  angelConnected: boolean;
  onRefresh: () => void;
}

export default function PortfolioPanel({ holdings, funds, prices, angelConnected, onRefresh }: PortfolioPanelProps) {
  const [selling, setSelling] = useState<string | null>(null);

  const totalInvested = holdings?.holdings.reduce((s, h) => s + h.qty * h.avg_price, 0) ?? 0;
  const liveValue     = holdings?.holdings.reduce((s, h) => s + h.qty * (prices[h.symbol] || h.ltp || h.avg_price), 0) ?? 0;
  const unrealizedPnl = liveValue - totalInvested;
  const cash          = funds?.funds.available ?? 0;
  const totalValue    = cash + liveValue;

  async function sellAll(sym: string) {
    setSelling(sym);
    try {
      const res = await fetch('/api/v1/wallet/trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: sym, action: 'sell' }),
      });
      if (res.ok) { toast.success(`Sold ${sym}`); onRefresh(); }
      else        { toast.error(`Failed to sell ${sym}`); }
    } catch { toast.error('Backend unreachable'); }
    finally { setSelling(null); }
  }

  if (!angelConnected) return (
    <div className="rounded-xl border border-border bg-card p-6 flex flex-col items-center justify-center gap-3 text-center min-h-[200px]">
      <Wallet className="h-8 w-8 text-muted-foreground/40" />
      <p className="text-sm text-muted-foreground">Angel One not configured</p>
      <p className="text-xs text-muted-foreground/60">Add credentials to .env and restart backend</p>
    </div>
  );

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      {/* Summary row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-border border-b border-border">
        {[
          { label: 'Total Value',    value: `₹${totalValue.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`, color: '' },
          { label: 'Cash',           value: `₹${cash.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`,       color: '' },
          { label: 'Invested',       value: `₹${totalInvested.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`, color: '' },
          { label: 'Unrealized P&L', value: `${unrealizedPnl >= 0 ? '+' : ''}₹${Math.abs(unrealizedPnl).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`,
            color: unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400' },
        ].map(item => (
          <div key={item.label} className="px-4 py-3">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wide">{item.label}</div>
            <div className={cn('text-base font-bold font-mono tabular-nums mt-0.5', item.color)}>{item.value}</div>
          </div>
        ))}
      </div>

      {/* Holdings table */}
      {holdings && holdings.holdings.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/20 border-b border-border">
              <tr>
                {['Symbol','Qty','Avg Price','LTP','Current Value','P&L','Action'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-xs font-medium text-muted-foreground uppercase tracking-wide whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {holdings.holdings.map(h => {
                const live    = prices[h.symbol] || h.ltp;
                const curVal  = h.qty * live;
                const livePnl = (live - h.avg_price) * h.qty;
                const livePct = ((live / h.avg_price) - 1) * 100;
                return (
                  <tr key={h.symbol} className="hover:bg-muted/10 transition-colors">
                    <td className="px-4 py-3 font-mono font-bold">{h.symbol}</td>
                    <td className="px-4 py-3 font-mono tabular-nums">{h.qty}</td>
                    <td className="px-4 py-3 font-mono text-muted-foreground tabular-nums">₹{h.avg_price.toFixed(2)}</td>
                    <td className="px-4 py-3 font-mono tabular-nums">
                      {live > 0 ? `₹${live.toFixed(2)}` : '—'}
                    </td>
                    <td className="px-4 py-3 font-mono tabular-nums">₹{curVal.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</td>
                    <td className="px-4 py-3">
                      <div className={cn('flex items-center gap-1 font-mono tabular-nums text-xs font-bold', livePnl >= 0 ? 'text-green-400' : 'text-red-400')}>
                        {livePnl >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                        {livePnl >= 0 ? '+' : ''}₹{Math.abs(livePnl).toFixed(0)}
                        <span className="text-[10px] opacity-70">({livePct >= 0 ? '+' : ''}{livePct.toFixed(2)}%)</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => sellAll(h.symbol)}
                        disabled={selling === h.symbol}
                        className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 text-xs font-medium transition-all disabled:opacity-50"
                      >
                        {selling === h.symbol ? <Loader2 className="h-3 w-3 animate-spin" /> : <Send className="h-3 w-3" />}
                        Sell
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-10 text-muted-foreground gap-2">
          <Wallet className="h-7 w-7 opacity-30" />
          <p className="text-sm">No open positions</p>
        </div>
      )}
    </div>
  );
}

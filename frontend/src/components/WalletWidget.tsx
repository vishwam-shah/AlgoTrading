'use client';
import { useState } from 'react';
import { Wallet, TrendingUp, TrendingDown, Plus, Minus, X, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

interface WalletWidgetProps {
  cash: number;               // available cash (Angel One or virtual)
  invested: number;           // current holdings value
  totalPnl: number;           // unrealized P&L
  holdings: number;           // count of positions
  angelConnected: boolean;
  onRefresh: () => void;
}

export default function WalletWidget({
  cash, invested, totalPnl, holdings, angelConnected, onRefresh
}: WalletWidgetProps) {
  const [open,    setOpen]    = useState(false);
  const [mode,    setMode]    = useState<'deposit'|'withdraw'|null>(null);
  const [amount,  setAmount]  = useState('');
  const [loading, setLoading] = useState(false);

  const totalValue = cash + invested;
  const pnlPct     = invested > 0 ? (totalPnl / invested) * 100 : 0;

  async function executeVirtualTrade(action: 'deposit' | 'withdraw') {
    const amt = parseFloat(amount);
    if (!amt || amt <= 0) { toast.error('Enter a valid amount'); return; }
    setLoading(true);
    try {
      // For virtual wallet: hit the v1 wallet reset/trade endpoint
      const res = await fetch('/api/v1/wallet/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ initial_balance: action === 'deposit' ? cash + amt : Math.max(0, cash - amt) }),
      });
      if (res.ok) {
        toast.success(`${action === 'deposit' ? 'Added' : 'Withdrawn'} ₹${amt.toLocaleString('en-IN')}`);
        setMode(null); setAmount(''); onRefresh();
      } else {
        toast.error('Failed — check backend');
      }
    } catch { toast.error('Backend unreachable'); }
    finally { setLoading(false); }
  }

  return (
    <div className="relative">
      {/* Trigger button */}
      <button
        onClick={() => setOpen(o => !o)}
        className={cn(
          'flex items-center gap-2 px-3 py-1.5 rounded-xl border transition-all text-sm',
          open ? 'bg-card border-primary/50 shadow-lg shadow-primary/5' : 'border-border bg-muted/20 hover:bg-muted/40'
        )}
      >
        <div className="flex items-center gap-1.5">
          <div className={cn('w-2 h-2 rounded-full', angelConnected ? 'bg-green-400' : 'bg-yellow-400')} />
          <Wallet className="h-3.5 w-3.5 text-muted-foreground" />
        </div>
        <div className="hidden sm:flex flex-col items-end">
          <span className="text-xs font-mono font-bold leading-tight">
            ₹{(totalValue / 100000).toFixed(1)}L
          </span>
          {totalPnl !== 0 && (
            <span className={cn('text-[9px] font-mono leading-tight', totalPnl >= 0 ? 'text-green-400' : 'text-red-400')}>
              {totalPnl >= 0 ? '+' : ''}₹{Math.abs(totalPnl).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
            </span>
          )}
        </div>
        <ChevronDown className={cn('h-3 w-3 text-muted-foreground transition-transform', open && 'rotate-180')} />
      </button>

      {/* Dropdown panel */}
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => { setOpen(false); setMode(null); }} />
          <div className="absolute right-0 top-full mt-2 w-72 z-50 rounded-2xl border border-border bg-card shadow-2xl shadow-black/20 overflow-hidden">

            {/* Header */}
            <div className="px-4 py-3 border-b border-border bg-gradient-to-r from-indigo-500/10 to-purple-500/10 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Wallet className="h-4 w-4 text-primary" />
                <span className="text-sm font-semibold">Portfolio Wallet</span>
              </div>
              <span className={cn('text-[10px] px-2 py-0.5 rounded-full border font-medium',
                angelConnected ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
              )}>
                {angelConnected ? 'Angel One' : 'Virtual'}
              </span>
            </div>

            {/* Stats */}
            <div className="p-4 space-y-3">
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: 'Total Value',  value: `₹${totalValue.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`, color: 'text-foreground' },
                  { label: 'Cash',         value: `₹${cash.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`,       color: 'text-foreground' },
                  { label: 'Invested',     value: `₹${invested.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`,   color: 'text-foreground' },
                  { label: 'Unrealized P&L', value: `${totalPnl >= 0 ? '+' : ''}₹${Math.abs(totalPnl).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`,
                    color: totalPnl > 0 ? 'text-green-400' : totalPnl < 0 ? 'text-red-400' : 'text-muted-foreground' },
                ].map(item => (
                  <div key={item.label} className="rounded-lg bg-muted/20 px-3 py-2">
                    <div className="text-[9px] text-muted-foreground uppercase tracking-wide">{item.label}</div>
                    <div className={cn('text-sm font-mono font-bold mt-0.5 tabular-nums', item.color)}>{item.value}</div>
                  </div>
                ))}
              </div>

              {/* P&L bar */}
              {invested > 0 && (
                <div>
                  <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                    <span>Return on invested</span>
                    <span className={cn('font-bold', pnlPct >= 0 ? 'text-green-400' : 'text-red-400')}>
                      {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%
                    </span>
                  </div>
                  <div className="w-full bg-muted/30 rounded-full h-1.5">
                    <div className={cn('h-1.5 rounded-full transition-all', totalPnl >= 0 ? 'bg-green-400' : 'bg-red-400')}
                      style={{ width: `${Math.min(Math.abs(pnlPct) * 5, 100)}%` }} />
                  </div>
                </div>
              )}

              <div className="text-xs text-muted-foreground flex justify-between">
                <span>{holdings} open positions</span>
                {angelConnected && <span className="text-green-400">Live data</span>}
              </div>
            </div>

            {/* Deposit / Withdraw (virtual wallet only) */}
            {!angelConnected && (
              <div className="px-4 pb-4 space-y-2">
                <div className="h-px bg-border" />
                {mode === null ? (
                  <div className="flex gap-2 pt-1">
                    <button onClick={() => setMode('deposit')}
                      className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg bg-green-500/10 hover:bg-green-500/20 text-green-400 border border-green-500/20 text-xs font-semibold transition-colors">
                      <Plus className="h-3.5 w-3.5" /> Add Funds
                    </button>
                    <button onClick={() => setMode('withdraw')}
                      className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 text-xs font-semibold transition-colors">
                      <Minus className="h-3.5 w-3.5" /> Withdraw
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2 pt-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium capitalize">{mode} Amount</span>
                      <button onClick={() => { setMode(null); setAmount(''); }}><X className="h-3.5 w-3.5 text-muted-foreground" /></button>
                    </div>
                    <div className="flex gap-2">
                      <div className="relative flex-1">
                        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">₹</span>
                        <input
                          type="number"
                          value={amount}
                          onChange={e => setAmount(e.target.value)}
                          placeholder="50000"
                          className="w-full pl-7 pr-3 py-2 rounded-lg border border-border bg-muted/20 text-sm font-mono focus:outline-none focus:border-primary/50"
                          onKeyDown={e => { if (e.key === 'Enter') executeVirtualTrade(mode); }}
                          autoFocus
                        />
                      </div>
                      <button
                        onClick={() => executeVirtualTrade(mode)}
                        disabled={loading || !amount}
                        className={cn(
                          'px-4 py-2 rounded-lg text-xs font-semibold transition-all disabled:opacity-50',
                          mode === 'deposit'
                            ? 'bg-green-500 hover:bg-green-400 text-white'
                            : 'bg-red-500 hover:bg-red-400 text-white'
                        )}>
                        {loading ? '…' : 'OK'}
                      </button>
                    </div>
                    {/* Quick amounts */}
                    <div className="flex gap-1.5">
                      {['25000','50000','100000','500000'].map(amt => (
                        <button key={amt} onClick={() => setAmount(amt)}
                          className="flex-1 py-1 rounded-md bg-muted/30 hover:bg-muted/60 text-[9px] font-mono transition-colors">
                          {parseInt(amt) >= 100000 ? `${parseInt(amt)/100000}L` : `${parseInt(amt)/1000}K`}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {angelConnected && (
              <div className="px-4 pb-4 pt-0">
                <div className="h-px bg-border mb-3" />
                <p className="text-[10px] text-muted-foreground text-center">
                  Funds managed via Angel One SmartAPI
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

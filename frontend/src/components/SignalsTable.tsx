'use client';
import { useState } from 'react';
import { TrendingUp, TrendingDown, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

interface Prediction {
  symbol: string;
  direction: 'UP' | 'DOWN';
  prob_up?: number;
  confidence?: number;
  ensemble_pred?: number;
}

interface SignalsTableProps {
  predictions: Prediction[];
  onSymbolClick?: (symbol: string) => void;
}

export default function SignalsTable({ predictions, onSymbolClick }: SignalsTableProps) {
  const [executing, setExecuting] = useState<string | null>(null);
  const [executed,  setExecuted]  = useState<Set<string>>(new Set());

  const upSignals = predictions
    .filter(p => p.direction === 'UP')
    .sort((a, b) => (b.prob_up ?? b.confidence ?? 0) - (a.prob_up ?? a.confidence ?? 0));

  const downSignals = predictions
    .filter(p => p.direction === 'DOWN')
    .sort((a, b) => (a.prob_up ?? a.confidence ?? 1) - (b.prob_up ?? b.confidence ?? 1));

  const sorted = [...upSignals, ...downSignals];

  async function executeTrade(sym: string) {
    setExecuting(sym);
    try {
      const res = await fetch('/api/v3/angel/execute-today?paper=true', {
        method: 'POST',
      });
      if (res.ok) {
        setExecuted(prev => new Set([...prev, sym]));
        toast.success(`Order queued for ${sym}`);
      } else {
        toast.error(`Failed to queue ${sym}`);
      }
    } catch {
      toast.error('Backend unreachable');
    } finally {
      setExecuting(null);
    }
  }

  if (!sorted.length) return (
    <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
      No predictions yet — pipeline still running
    </div>
  );

  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/30 border-b border-border">
          <tr>
            <th className="text-left px-4 py-3 font-medium">Symbol</th>
            <th className="text-center px-4 py-3 font-medium">Signal</th>
            <th className="text-right px-4 py-3 font-medium">P(UP)</th>
            <th className="text-right px-4 py-3 font-medium">Confidence</th>
            <th className="text-right px-4 py-3 font-medium">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/40">
          {sorted.map((p, i) => {
            const conf   = p.prob_up ?? p.confidence ?? 0.5;
            const isUp   = p.direction === 'UP';
            const strong = conf >= 0.58;
            return (
              <tr
                key={p.symbol}
                className={cn(
                  'hover:bg-muted/10 transition-colors',
                  i % 2 === 0 ? '' : 'bg-muted/5'
                )}
              >
                <td className="px-4 py-3">
                  <button
                    onClick={() => onSymbolClick?.(p.symbol)}
                    className="font-mono font-bold hover:text-primary transition-colors"
                  >
                    {p.symbol}
                  </button>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={cn(
                    'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold',
                    isUp
                      ? 'bg-green-500/15 text-green-400 border border-green-500/20'
                      : 'bg-red-500/15 text-red-400 border border-red-500/20'
                  )}>
                    {isUp ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                    {p.direction}
                  </span>
                </td>
                <td className="px-4 py-3 text-right font-mono">
                  <span className={cn(
                    'text-sm font-bold',
                    conf >= 0.58 ? 'text-green-400'
                    : conf >= 0.52 ? 'text-yellow-400'
                    : 'text-muted-foreground'
                  )}>
                    {(conf * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="px-4 py-3 text-right">
                  <div className="w-full bg-muted/30 rounded-full h-1.5 max-w-[80px] ml-auto">
                    <div
                      className={cn('h-1.5 rounded-full', isUp ? 'bg-green-500' : 'bg-red-500')}
                      style={{ width: `${Math.min(conf * 100 * 1.5, 100)}%` }}
                    />
                  </div>
                </td>
                <td className="px-4 py-3 text-right">
                  {isUp ? (
                    executed.has(p.symbol) ? (
                      <span className="flex items-center justify-end gap-1 text-xs text-green-400">
                        <CheckCircle2 className="h-3.5 w-3.5" /> Queued
                      </span>
                    ) : (
                      <button
                        onClick={() => executeTrade(p.symbol)}
                        disabled={executing === p.symbol}
                        className={cn(
                          'px-3 py-1 rounded-lg text-xs font-semibold transition-all',
                          strong
                            ? 'bg-green-500 hover:bg-green-400 text-white'
                            : 'bg-muted hover:bg-muted/80 text-foreground border border-border'
                        )}
                      >
                        {executing === p.symbol
                          ? <Loader2 className="h-3 w-3 animate-spin" />
                          : 'Execute'}
                      </button>
                    )
                  ) : (
                    <span className="text-xs text-muted-foreground">—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

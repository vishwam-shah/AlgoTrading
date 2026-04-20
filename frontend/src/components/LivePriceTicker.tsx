'use client';
import { TrendingUp, TrendingDown, Wifi, WifiOff } from 'lucide-react';
import { cn } from '@/lib/utils';

const DEFAULT_SYMBOLS = [
  'SBIN','HDFCBANK','ICICIBANK','TCS','INFY','RELIANCE',
  'AXISBANK','KOTAKBANK','WIPRO','HCLTECH',
];

interface Props {
  symbols?: string[];
  prices?: Record<string, number>;
  change?: (sym: string) => number | null;
  connected?: boolean;
}

export default function LivePriceTicker({
  symbols = DEFAULT_SYMBOLS,
  prices = {},
  change,
  connected = false,
}: Props) {
  return (
    <div className="w-full">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Live</span>
        <span className={cn('flex items-center gap-1 text-[10px]', connected ? 'text-green-400' : 'text-muted-foreground')}>
          {connected ? <Wifi className="h-2.5 w-2.5" /> : <WifiOff className="h-2.5 w-2.5" />}
          {connected ? 'connected' : 'connecting…'}
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        {symbols.map(sym => {
          const price = prices[sym];
          const chg   = change ? change(sym) : null;
          const up    = chg !== null && chg !== undefined && chg >= 0;

          return (
            <div key={sym} className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border bg-card hover:bg-muted/20 transition-colors">
              <span className="text-xs font-mono font-bold">{sym}</span>
              {price ? (
                <>
                  <span className="text-sm font-mono font-bold tabular-nums">
                    ₹{price.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                  </span>
                  {chg !== null && chg !== undefined && (
                    <span className={cn('flex items-center gap-0.5 text-[10px] font-semibold tabular-nums', up ? 'text-green-400' : 'text-red-400')}>
                      {up ? <TrendingUp className="h-2.5 w-2.5" /> : <TrendingDown className="h-2.5 w-2.5" />}
                      {chg >= 0 ? '+' : ''}{chg.toFixed(2)}%
                    </span>
                  )}
                </>
              ) : (
                <span className="text-xs text-muted-foreground">—</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

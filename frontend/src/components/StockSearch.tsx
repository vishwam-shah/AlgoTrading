'use client';
import { useState, useEffect, useRef } from 'react';
import { Search, TrendingUp, TrendingDown, X, Star, StarOff } from 'lucide-react';
import { cn } from '@/lib/utils';
import { apiFetch } from '@/hooks/useData';

interface StockResult {
  symbol: string;
  oos_accuracy?: number;
  direction?: string;
  prob_up?: number;
  n_predictions?: number;
  best_model?: string;
  avg_lgbm_acc?: number;
  avg_xgb_acc?: number;
}

interface StockSearchProps {
  allStocks: StockResult[];
  onSelect: (symbol: string) => void;
  watchlist: string[];
  onWatchlistToggle: (symbol: string) => void;
}

export default function StockSearch({ allStocks, onSelect, watchlist, onWatchlistToggle }: StockSearchProps) {
  const [query, setQuery]       = useState('');
  const [open, setOpen]         = useState(false);
  const [focused, setFocused]   = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef  = useRef<HTMLDivElement>(null);

  const results = query.length >= 1
    ? allStocks.filter(s =>
        s.symbol.toLowerCase().includes(query.toLowerCase())
      ).slice(0, 10)
    : allStocks
        .filter(s => watchlist.includes(s.symbol))
        .slice(0, 10);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault(); inputRef.current?.focus(); setOpen(true);
      }
      if (e.key === 'Escape') { setOpen(false); setQuery(''); }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  function select(sym: string) {
    onSelect(sym); setQuery(''); setOpen(false);
  }

  return (
    <div className="relative w-full max-w-sm">
      <div className={cn(
        'flex items-center gap-2 px-3 py-2 rounded-xl border transition-all',
        open ? 'border-primary/50 ring-2 ring-primary/10 bg-card' : 'border-border bg-muted/20 hover:border-border/80'
      )}>
        <Search className="h-4 w-4 text-muted-foreground flex-shrink-0" />
        <input
          ref={inputRef}
          value={query}
          onChange={e => { setQuery(e.target.value); setOpen(true); }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          placeholder="Search stocks… (⌘K)"
          className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground/60"
          onKeyDown={e => {
            if (e.key === 'ArrowDown') setFocused(f => Math.min(f + 1, results.length - 1));
            if (e.key === 'ArrowUp')   setFocused(f => Math.max(f - 1, 0));
            if (e.key === 'Enter' && focused >= 0) select(results[focused].symbol);
          }}
        />
        {query && <button onClick={() => setQuery('')}><X className="h-3.5 w-3.5 text-muted-foreground" /></button>}
      </div>

      {open && results.length > 0 && (
        <div
          ref={listRef}
          className="absolute top-full mt-1 w-full z-50 rounded-xl border border-border bg-popover shadow-xl overflow-hidden"
        >
          {!query && watchlist.length > 0 && (
            <div className="px-3 py-1.5 text-[10px] text-muted-foreground uppercase tracking-widest border-b border-border">Watchlist</div>
          )}
          {results.map((s, i) => {
            const up   = s.direction === 'UP' || (s.prob_up ?? 0) >= 0.52;
            const acc  = s.oos_accuracy ?? 0;
            const inWL = watchlist.includes(s.symbol);
            return (
              <div
                key={s.symbol}
                className={cn(
                  'flex items-center gap-3 px-3 py-2.5 cursor-pointer transition-colors',
                  focused === i ? 'bg-muted' : 'hover:bg-muted/50'
                )}
                onMouseDown={() => select(s.symbol)}
                onMouseEnter={() => setFocused(i)}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-sm">{s.symbol}</span>
                    {s.direction && (
                      <span className={cn(
                        'flex items-center gap-0.5 text-[10px] font-semibold px-1.5 py-0.5 rounded-full',
                        up ? 'bg-green-500/15 text-green-400' : 'bg-red-500/15 text-red-400'
                      )}>
                        {up ? <TrendingUp className="h-2.5 w-2.5" /> : <TrendingDown className="h-2.5 w-2.5" />}
                        {s.direction}
                      </span>
                    )}
                  </div>
                  <div className="text-[10px] text-muted-foreground mt-0.5">
                    {acc > 0 && `OOS: ${(acc * 100).toFixed(1)}%`}
                    {s.best_model && ` · Best: ${s.best_model}`}
                    {s.n_predictions && ` · ${s.n_predictions} predictions`}
                  </div>
                </div>

                {s.prob_up != null && (
                  <div className={cn('text-xs font-mono font-bold tabular-nums', s.prob_up >= 0.52 ? 'text-green-400' : 'text-muted-foreground')}>
                    {(s.prob_up * 100).toFixed(1)}%
                  </div>
                )}

                <button
                  onMouseDown={e => { e.stopPropagation(); onWatchlistToggle(s.symbol); }}
                  className="text-muted-foreground hover:text-yellow-400 transition-colors"
                >
                  {inWL ? <Star className="h-3.5 w-3.5 fill-yellow-400 text-yellow-400" /> : <StarOff className="h-3.5 w-3.5" />}
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

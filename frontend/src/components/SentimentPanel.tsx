'use client';
import { useEffect, useState } from 'react';
import { Newspaper, TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { apiFetch } from '@/hooks/useData';

interface SentRow {
  symbol: string;
  raw_score: number;
  n_articles: number;
  positive_ratio: number;
  negative_ratio: number;
}

interface TradeableRow {
  symbol: string;
  direction: string;
  confidence: number;
  sentiment_score: number;
  sentiment_prob: number;
  n_articles: number;
  positive_ratio: number;
  negative_ratio: number;
}

interface SentimentOverview {
  latest_date: string;
  n_symbols: number;
  n_zero_coverage: number;
  avg_score: number;
  avg_articles: number;
  model_used: string;
  top_bullish: SentRow[];
  top_bearish: SentRow[];
  tradeable: TradeableRow[];
  all_scores: SentRow[];
  blend_weight: number;
}

function scoreBar(score: number) {
  const pct = Math.min(100, Math.abs(score) * 200);
  const isPos = score >= 0;
  return (
    <div className="flex items-center gap-1.5 w-32">
      <div className="flex-1 h-1.5 bg-muted/30 rounded-full relative overflow-hidden">
        <div className={cn(
          'absolute h-full rounded-full',
          isPos ? 'bg-green-500 left-1/2' : 'bg-red-500 right-1/2'
        )} style={{ width: `${pct / 2}%` }} />
        <div className="absolute left-1/2 top-0 h-full w-px bg-border" />
      </div>
      <span className={cn('font-mono text-[10px] tabular-nums w-12 text-right',
        isPos ? 'text-green-400' : 'text-red-400')}>
        {isPos ? '+' : ''}{score.toFixed(2)}
      </span>
    </div>
  );
}

function MetricCard({ label, value, sub, color = '' }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="rounded-xl border border-border bg-card p-3">
      <div className="text-[10px] text-muted-foreground uppercase tracking-wide">{label}</div>
      <div className={cn('text-lg font-bold font-mono mt-1 tabular-nums', color)}>{value}</div>
      {sub && <div className="text-[10px] text-muted-foreground mt-0.5">{sub}</div>}
    </div>
  );
}

export default function SentimentPanel() {
  const [data, setData] = useState<SentimentOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    apiFetch<SentimentOverview>('/api/v3/sentiment/overview')
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="rounded-xl border border-border bg-card p-8 text-center text-muted-foreground text-sm">
      Loading sentiment data…
    </div>
  );

  if (!data) return (
    <div className="rounded-xl border border-border bg-card p-8 flex flex-col items-center gap-2 text-muted-foreground">
      <Newspaper className="h-7 w-7 opacity-30" />
      <p className="text-sm">No sentiment data yet</p>
      <p className="text-xs opacity-60">Run <code className="font-mono bg-muted px-1 rounded">V3/01_data/news/sentiment_history.py</code></p>
    </div>
  );

  const coveragePct = data.n_symbols > 0 ? ((data.n_symbols - data.n_zero_coverage) / data.n_symbols) * 100 : 0;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <Newspaper className="h-4 w-4 text-primary" /> News Sentiment (FinBERT)
          <span className="text-[10px] text-muted-foreground font-normal ml-1">
            {data.model_used} · {data.latest_date} · blend {(data.blend_weight * 100).toFixed(0)}%
          </span>
        </h3>

        {/* Summary metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-5 gap-3">
          <MetricCard label="Coverage" value={`${data.n_symbols - data.n_zero_coverage}/${data.n_symbols}`}
            sub={`${coveragePct.toFixed(0)}% symbols have articles`}
            color={coveragePct >= 90 ? 'text-green-400' : coveragePct >= 70 ? 'text-yellow-400' : 'text-red-400'} />
          <MetricCard label="Avg Score" value={(data.avg_score >= 0 ? '+' : '') + data.avg_score.toFixed(3)}
            sub="mean across all symbols"
            color={data.avg_score > 0.05 ? 'text-green-400' : data.avg_score < -0.05 ? 'text-red-400' : 'text-muted-foreground'} />
          <MetricCard label="Avg Articles" value={data.avg_articles.toFixed(1)}
            sub="per symbol / day" />
          <MetricCard label="Bullish Bias" value={`${data.top_bullish.length}`}
            sub="stocks with score > 0" color="text-green-400" />
          <MetricCard label="Bearish Bias" value={`${data.top_bearish.length}`}
            sub="stocks with score < 0" color="text-red-400" />
        </div>
      </div>

      {/* Top bullish / bearish */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="px-4 py-2.5 border-b border-border bg-green-500/5 flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-green-400" />
            <span className="text-sm font-semibold">Top 5 Bullish</span>
          </div>
          <div className="divide-y divide-border/30">
            {data.top_bullish.map(r => (
              <div key={r.symbol} className="px-4 py-2 flex items-center justify-between hover:bg-muted/10">
                <div className="flex items-center gap-3 min-w-0">
                  <span className="font-mono font-bold text-sm w-24 truncate">{r.symbol}</span>
                  <span className="text-[10px] text-muted-foreground">{r.n_articles} articles</span>
                </div>
                {scoreBar(r.raw_score)}
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="px-4 py-2.5 border-b border-border bg-red-500/5 flex items-center gap-2">
            <TrendingDown className="h-4 w-4 text-red-400" />
            <span className="text-sm font-semibold">Top 5 Bearish</span>
          </div>
          <div className="divide-y divide-border/30">
            {data.top_bearish.map(r => (
              <div key={r.symbol} className="px-4 py-2 flex items-center justify-between hover:bg-muted/10">
                <div className="flex items-center gap-3 min-w-0">
                  <span className="font-mono font-bold text-sm w-24 truncate">{r.symbol}</span>
                  <span className="text-[10px] text-muted-foreground">{r.n_articles} articles</span>
                </div>
                {scoreBar(r.raw_score)}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tradeable stocks with sentiment impact */}
      {data.tradeable.length > 0 && (
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold">Sentiment Impact on Tradeable Signals</span>
            </div>
            <span className="text-[10px] text-muted-foreground">
              final = 0.80·ensemble + {data.blend_weight.toFixed(2)}·sentiment
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-muted/10 text-muted-foreground">
                <tr>
                  {['Symbol','Signal','Ensemble Conf','News Score','News→Prob','Articles','Pos/Neg'].map(h =>
                    <th key={h} className="text-left px-3 py-2 font-medium whitespace-nowrap">{h}</th>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {data.tradeable.map(r => (
                  <tr key={r.symbol} className="hover:bg-muted/10">
                    <td className="px-3 py-2 font-mono font-bold">{r.symbol}</td>
                    <td className="px-3 py-2">
                      <span className={cn('font-mono px-1.5 py-0.5 rounded text-[10px]',
                        r.direction === 'UP' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                      )}>{r.direction}</span>
                    </td>
                    <td className="px-3 py-2 font-mono">{(r.confidence * 100).toFixed(1)}%</td>
                    <td className={cn('px-3 py-2 font-mono',
                      r.sentiment_score > 0.05 ? 'text-green-400' : r.sentiment_score < -0.05 ? 'text-red-400' : 'text-muted-foreground'
                    )}>
                      {r.sentiment_score >= 0 ? '+' : ''}{r.sentiment_score.toFixed(3)}
                    </td>
                    <td className="px-3 py-2 font-mono">{(r.sentiment_prob * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 font-mono text-muted-foreground">
                      {r.n_articles === 0 ? (
                        <span className="inline-flex items-center gap-1 text-yellow-500">
                          <AlertCircle className="h-3 w-3" />0
                        </span>
                      ) : r.n_articles}
                    </td>
                    <td className="px-3 py-2 font-mono text-[10px]">
                      <span className="text-green-400">{(r.positive_ratio * 100).toFixed(0)}%</span>
                      <span className="text-muted-foreground mx-1">/</span>
                      <span className="text-red-400">{(r.negative_ratio * 100).toFixed(0)}%</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* All stocks full table (collapsible) */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <button onClick={() => setShowAll(!showAll)}
          className="w-full px-4 py-3 border-b border-border flex items-center justify-between hover:bg-muted/10">
          <span className="text-sm font-semibold">All {data.n_symbols} Symbols</span>
          <span className="text-xs text-muted-foreground">{showAll ? 'hide' : 'show'} · sorted by score</span>
        </button>
        {showAll && (
          <div className="max-h-96 overflow-y-auto">
            <table className="w-full text-xs">
              <tbody className="divide-y divide-border/30">
                {data.all_scores.map(r => (
                  <tr key={r.symbol} className="hover:bg-muted/10">
                    <td className="px-3 py-1.5 font-mono font-bold w-32">{r.symbol}</td>
                    <td className="px-3 py-1.5">{scoreBar(r.raw_score)}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-[10px] text-muted-foreground w-20">
                      {r.n_articles} art.
                    </td>
                    <td className="px-3 py-1.5 font-mono text-[10px] w-24">
                      <span className="text-green-400">{(r.positive_ratio * 100).toFixed(0)}%</span>
                      <span className="text-muted-foreground mx-1">/</span>
                      <span className="text-red-400">{(r.negative_ratio * 100).toFixed(0)}%</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

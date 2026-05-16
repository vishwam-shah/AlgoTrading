'use client';
import { TrendingUp, TrendingDown, BarChart2, FlaskConical } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SummaryStock {
  symbol: string;
  oos_accuracy: number;
  oos_f1?: number;
  n_predictions: number;
  n_windows?: number;
  n_features?: number;
  best_model?: string;
}

interface Props {
  summary: { stocks: SummaryStock[] } | null;
  capital: number;
}

const SECTORS: Record<string, string[]> = {
  IT:       ['TCS','INFY','HCLTECH','WIPRO','TECHM','LTIM','MPHASIS','PERSISTENT','COFORGE','TATAELXSI','OFSS','NAUKRI'],
  Banking:  ['SBIN','HDFCBANK','ICICIBANK','AXISBANK','KOTAKBANK','INDUSINDBK','BANDHANBNK','IDFCFIRSTB','FEDERALBNK','AUBANK','RBLBANK'],
  Finance:  ['BAJFINANCE','BAJAJFINSV','HDFCLIFE','SBILIFE','ICICIGI','MUTHOOTFIN','CHOLAFIN','SHRIRAMFIN','MANAPPURAM','LICHSGFIN'],
  FMCG:     ['HINDUNILVR','ITC','NESTLEIND','BRITANNIA','TATACONSUM','MARICO','COLPAL','GODREJCP'],
  Auto:     ['MARUTI','BAJAJ-AUTO','HEROMOTOCO','EICHERMOT','TVSMOTOR','M&M','MOTHERSON','BOSCHLTD','EXIDEIND'],
  Pharma:   ['SUNPHARMA','DRREDDY','CIPLA','DIVISLAB','LUPIN','TORNTPHARM','AUROPHARMA','ALKEM'],
  Energy:   ['RELIANCE','ONGC','BPCL','NTPC','POWERGRID','COALINDIA','GAIL','TATAPOWER'],
  Metals:   ['TATASTEEL','HINDALCO','JSWSTEEL','VEDL','SAIL','NMDC'],
  Infra:    ['LT','BHEL','SIEMENS','ABB','HAVELLS','POLYCAB','CUMMINSIND','BHARTIARTL','INDUSTOWER'],
  Cement:   ['ULTRACEMCO','GRASIM','AMBUJACEM','SHREECEM'],
  Consumer: ['TITAN','ASIANPAINT','PIDILITIND','BERGEPAINT','VOLTAS','PAGEIND'],
  Realty:   ['DLF','DMART','GODREJPROP','ADANIENT','ADANIPORTS'],
  Defence:  ['BEL','HAL','IRFC','ETERNAL'],
};

function MetricCard({ label, value, sub, color = '', badge }: {
  label: string; value: string; sub?: string; color?: string; badge?: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className="flex items-start justify-between">
        <div className="text-[10px] text-muted-foreground uppercase tracking-wide">{label}</div>
        {badge && <span className="text-[9px] px-1.5 py-0.5 rounded bg-muted/40 text-muted-foreground font-mono">{badge}</span>}
      </div>
      <div className={cn('text-xl font-bold font-mono mt-1 tabular-nums', color)}>{value}</div>
      {sub && <div className="text-xs text-muted-foreground mt-0.5">{sub}</div>}
    </div>
  );
}

function AccBar({ value, max = 100, color }: { value: number; max?: number; color: string }) {
  return (
    <div className="w-full bg-muted/30 rounded-full h-1.5">
      <div className={cn('h-1.5 rounded-full transition-all', color)} style={{ width: `${Math.min((value / max) * 100, 100)}%` }} />
    </div>
  );
}

export default function OOSMetricsPanel({ summary }: Props) {
  const stocks    = summary?.stocks?.filter(s => s.symbol !== 'AVERAGE') ?? [];
  const nTrained  = stocks.length;
  const avgAcc    = nTrained > 0 ? stocks.reduce((s, r) => s + (r.oos_accuracy ?? 0), 0) / nTrained : 0;
  const avgF1     = nTrained > 0 ? stocks.reduce((s, r) => s + (r.oos_f1 ?? 0), 0) / nTrained : 0;
  const liftVsRandom = avgAcc / 0.20;

  const above58 = stocks.filter(s => (s.oos_accuracy ?? 0) >= 0.58).length;
  const above54 = stocks.filter(s => (s.oos_accuracy ?? 0) >= 0.54).length;
  const above50 = stocks.filter(s => (s.oos_accuracy ?? 0) >= 0.50).length;
  const below50 = nTrained - above50;
  const below45 = stocks.filter(s => (s.oos_accuracy ?? 0) < 0.45).length;

  const sorted  = [...stocks].sort((a, b) => (b.oos_accuracy ?? 0) - (a.oos_accuracy ?? 0));
  const top5    = sorted.slice(0, 5);
  const bottom5 = sorted.slice(-5).reverse();

  const sectorStats = Object.entries(SECTORS).map(([name, syms]) => {
    const s = stocks.filter(st => syms.includes(st.symbol));
    if (!s.length) return null;
    const avg = s.reduce((a, b) => a + (b.oos_accuracy ?? 0), 0) / s.length;
    return { name, avg, count: s.length };
  }).filter(Boolean) as { name: string; avg: number; count: number }[];
  sectorStats.sort((a, b) => b.avg - a.avg);

  const buckets = [
    { label: '<45%',   count: stocks.filter(s => (s.oos_accuracy ?? 0) < 0.45).length,                                                        color: 'bg-red-500' },
    { label: '45–50%', count: stocks.filter(s => (s.oos_accuracy ?? 0) >= 0.45 && (s.oos_accuracy ?? 0) < 0.50).length,                       color: 'bg-orange-500' },
    { label: '50–52%', count: stocks.filter(s => (s.oos_accuracy ?? 0) >= 0.50 && (s.oos_accuracy ?? 0) < 0.52).length,                       color: 'bg-yellow-500' },
    { label: '52–54%', count: stocks.filter(s => (s.oos_accuracy ?? 0) >= 0.52 && (s.oos_accuracy ?? 0) < 0.54).length,                       color: 'bg-emerald-400' },
    { label: '54–58%', count: stocks.filter(s => (s.oos_accuracy ?? 0) >= 0.54 && (s.oos_accuracy ?? 0) < 0.58).length,                       color: 'bg-green-400' },
    { label: '≥58%',   count: stocks.filter(s => (s.oos_accuracy ?? 0) >= 0.58).length,                                                        color: 'bg-green-300' },
  ];
  const maxBucket = Math.max(...buckets.map(b => b.count), 1);

  return (
    <div className="space-y-6">
      {/* Header metrics */}
      <div>
        <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <BarChart2 className="h-4 w-4 text-primary" /> OOS Model Performance
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
          <MetricCard label="Stocks Trained"  value={`${nTrained}`} sub="walk-forward" />
          <MetricCard label="Avg OOS Accuracy" value={`${(avgAcc * 100).toFixed(2)}%`}
            sub="5-class, walk-forward" badge="acc"
            color={avgAcc >= 0.54 ? 'text-green-400' : avgAcc >= 0.50 ? 'text-yellow-400' : 'text-red-400'} />
          <MetricCard label="Avg OOS F1" value={`${(avgF1 * 100).toFixed(2)}%`}
            sub="macro weighted" badge="f1"
            color={avgF1 >= 0.58 ? 'text-green-400' : avgF1 >= 0.54 ? 'text-yellow-400' : ''} />
          <MetricCard label="Lift vs Random" value={`${liftVsRandom.toFixed(2)}×`}
            sub="vs 20% baseline"
            color={liftVsRandom >= 2.5 ? 'text-green-400' : 'text-yellow-400'} />
          <MetricCard label="Above 54%" value={`${above54}`} sub="publication quality"
            color={above54 >= 10 ? 'text-green-400' : above54 >= 5 ? 'text-yellow-400' : 'text-muted-foreground'} />
          <MetricCard label="Above 50%" value={`${above50}`} sub="statistically valid" />
          <MetricCard label="Below 50%" value={`${below50}`}
            sub={below45 > 0 ? `${below45} below 45%` : 'needs attention'}
            color={below50 > 40 ? 'text-red-400' : below50 > 25 ? 'text-yellow-400' : ''} />
        </div>
      </div>

      {/* Distribution + Benchmarks */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-xl border border-border bg-card p-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-4">
            Accuracy Distribution ({nTrained} stocks)
          </h4>
          <div className="flex items-end gap-2 h-20">
            {buckets.map(b => (
              <div key={b.label} className="flex-1 flex flex-col items-center gap-1">
                <span className="text-[10px] font-mono font-bold text-muted-foreground">{b.count}</span>
                <div className="w-full flex items-end justify-center" style={{ height: 48 }}>
                  <div className={cn('w-full rounded-t-sm transition-all', b.color, b.count === 0 ? 'opacity-20' : '')}
                    style={{ height: `${Math.max((b.count / maxBucket) * 48, b.count > 0 ? 4 : 0)}px` }} />
                </div>
                <span className="text-[9px] text-muted-foreground text-center leading-tight">{b.label}</span>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-3 border-t border-border grid grid-cols-3 gap-2 text-[10px] text-muted-foreground">
            <div>Min <span className="font-mono text-foreground">{(Math.min(...stocks.map(s => s.oos_accuracy ?? 0)) * 100).toFixed(1)}%</span></div>
            <div className="text-center">Median <span className="font-mono text-foreground">{((sorted[Math.floor(sorted.length / 2)]?.oos_accuracy ?? 0) * 100).toFixed(1)}%</span></div>
            <div className="text-right">Max <span className="font-mono text-foreground">{(Math.max(...stocks.map(s => s.oos_accuracy ?? 0)) * 100).toFixed(1)}%</span></div>
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-4">
            Accuracy vs Benchmarks
          </h4>
          <div className="space-y-3">
            {[
              { label: 'Random 5-class baseline', val: 20.0,         color: 'bg-red-500',     note: 'Coin-flip equivalent' },
              { label: 'Our avg OOS accuracy',    val: avgAcc * 100, color: 'bg-indigo-500',  note: `${liftVsRandom.toFixed(2)}× lift vs random` },
              { label: 'Publication bar (52%)',   val: 52.0,         color: 'bg-yellow-500',  note: 'Min for research paper' },
              { label: 'Trading edge (54%)',       val: 54.0,         color: 'bg-green-500',   note: 'Statistically profitable' },
              { label: 'Strong edge (58%)',        val: 58.0,         color: 'bg-emerald-400', note: `Only ${above58} stocks reach this` },
            ].map(b => (
              <div key={b.label}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted-foreground">{b.label}</span>
                  <span className={cn('font-mono font-bold',
                    b.label.startsWith('Our') ? (avgAcc >= 0.54 ? 'text-green-400' : avgAcc >= 0.50 ? 'text-yellow-400' : 'text-red-400') : ''
                  )}>{b.val.toFixed(1)}%</span>
                </div>
                <AccBar value={b.val} color={b.label.startsWith('Our') && avgAcc < 0.52 ? 'bg-yellow-500' : b.color} />
                <div className="text-[9px] text-muted-foreground mt-0.5">{b.note}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Sector breakdown */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-primary" />
          <span className="text-sm font-semibold">Sector-wise OOS Accuracy</span>
          <span className="text-xs text-muted-foreground ml-auto">Target: ≥52%</span>
        </div>
        <div className="p-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {sectorStats.map(s => {
            const pct = s.avg * 100;
            return (
              <div key={s.name} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="font-medium">{s.name}</span>
                  <span className={cn('font-mono font-bold',
                    pct >= 54 ? 'text-green-400' : pct >= 52 ? 'text-emerald-400' : pct >= 50 ? 'text-yellow-400' : 'text-red-400'
                  )}>{pct.toFixed(1)}%</span>
                </div>
                <AccBar value={pct} max={60}
                  color={pct >= 54 ? 'bg-green-500' : pct >= 52 ? 'bg-emerald-400' : pct >= 50 ? 'bg-yellow-400' : 'bg-red-400'} />
                <div className="text-[9px] text-muted-foreground">{s.count} stocks</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Top + Bottom */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2 bg-green-500/5">
            <TrendingUp className="h-4 w-4 text-green-400" />
            <span className="text-sm font-semibold text-green-400">Top 5 — Trade these</span>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-muted/10 text-xs text-muted-foreground">
              <tr>{['Symbol','OOS Acc','F1','Predictions'].map(h =>
                <th key={h} className="text-left px-4 py-2 font-medium">{h}</th>)}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {top5.map(s => (
                <tr key={s.symbol} className="hover:bg-muted/10">
                  <td className="px-4 py-2.5 font-mono font-bold">{s.symbol}</td>
                  <td className="px-4 py-2.5">
                    <span className={cn('font-mono font-bold',
                      (s.oos_accuracy ?? 0) >= 0.54 ? 'text-green-400' : 'text-yellow-400'
                    )}>{((s.oos_accuracy ?? 0) * 100).toFixed(1)}%</span>
                  </td>
                  <td className="px-4 py-2.5 font-mono text-muted-foreground">
                    {s.oos_f1 ? `${(s.oos_f1 * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-muted-foreground">{s.n_predictions}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2 bg-red-500/5">
            <TrendingDown className="h-4 w-4 text-red-400" />
            <span className="text-sm font-semibold text-red-400">Bottom 5 — Exclude from trading</span>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-muted/10 text-xs text-muted-foreground">
              <tr>{['Symbol','OOS Acc','F1','Predictions'].map(h =>
                <th key={h} className="text-left px-4 py-2 font-medium">{h}</th>)}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {bottom5.map(s => (
                <tr key={s.symbol} className="hover:bg-muted/10">
                  <td className="px-4 py-2.5 font-mono font-bold">{s.symbol}</td>
                  <td className="px-4 py-2.5">
                    <span className={cn('font-mono font-bold',
                      (s.oos_accuracy ?? 0) < 0.47 ? 'text-red-400' : 'text-orange-400'
                    )}>{((s.oos_accuracy ?? 0) * 100).toFixed(1)}%</span>
                  </td>
                  <td className="px-4 py-2.5 font-mono text-muted-foreground">
                    {s.oos_f1 ? `${(s.oos_f1 * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-muted-foreground">{s.n_predictions}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

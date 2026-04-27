'use client';
import { cn } from '@/lib/utils';

// Only active models: LSTM/GRU/CNN-* were dropped (underperformed < 51.5%)
// Full mode: LGB + XGB + BiLSTM + TCN-T + NBEATS → ensemble
// Fast mode: LGB + XGB only → ensemble
const MODELS = [
  { key: 'avg_lgbm_acc',            label: 'LGB',    group: 'tree', always: true },
  { key: 'avg_xgb_acc',             label: 'XGB',    group: 'tree', always: true },
  { key: 'avg_bilstm_acc',          label: 'BiLSTM', group: 'dl',   always: false },
  { key: 'avg_tcn_transformer_acc', label: 'TCN-T',  group: 'dl',   always: false },
  { key: 'avg_nbeats_acc',          label: 'NBEATS', group: 'dl',   always: false },
  { key: 'oos_accuracy',            label: 'ENS ★',  group: 'ens',  always: true },
];

function cellColor(val: number | null, isEns = false) {
  if (val === null) return 'text-muted-foreground/20';
  if (val >= 0.58) return isEns ? 'bg-green-500/90 text-white font-bold' : 'bg-green-500/70 text-white font-semibold';
  if (val >= 0.54) return isEns ? 'bg-green-500/50 text-green-200 font-semibold' : 'bg-green-500/30 text-green-300';
  if (val >= 0.52) return isEns ? 'bg-emerald-500/30 text-emerald-300 font-medium' : 'bg-emerald-500/15 text-emerald-400';
  if (val >= 0.50) return isEns ? 'bg-yellow-500/35 text-yellow-200 font-medium' : 'bg-yellow-500/15 text-yellow-300';
  return isEns ? 'bg-red-500/30 text-red-300' : 'bg-red-500/10 text-red-400/70';
}

function getVal(row: any, key: string): number | null {
  const v = row[key];
  if (v === undefined || v === null || v === '—') return null;
  const n = parseFloat(v);
  return isNaN(n) || n === 0 ? null : n;
}

function hasData(stocks: any[], keys: string[]): boolean {
  const sample = stocks.find(s => s.symbol !== 'AVERAGE');
  if (!sample) return false;
  return keys.some(k => {
    const v = sample[k];
    return v !== undefined && v !== null && v !== 0 && v !== '—' && parseFloat(v) > 0;
  });
}

export default function AccuracyHeatmap({ stocks }: { stocks: any[] }) {
  const rows = stocks.filter(s => s.symbol !== 'AVERAGE');

  const hasDL   = hasData(rows, MODELS.filter(m => m.group === 'dl').map(m => m.key));
  const hasTree = hasData(rows, MODELS.filter(m => m.group === 'tree').map(m => m.key));

  // Show only columns that have data for this run
  const cols = MODELS.filter(m => {
    if (m.group === 'ens')  return true;
    if (m.group === 'tree') return hasTree;
    if (m.group === 'dl')   return hasDL;
    return false;
  });

  const dlCount  = hasDL ? 3 : 0;
  const treeCount = hasTree ? 2 : 0;

  // Per-column average (exclude 0/null)
  const colAvg = (key: string) => {
    const vals = rows.map(r => getVal(r, key)).filter((v): v is number => v !== null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };

  const above54 = rows.filter(r => (r.oos_accuracy ?? 0) >= 0.54).length;
  const above50 = rows.filter(r => (r.oos_accuracy ?? 0) >= 0.50 && (r.oos_accuracy ?? 0) < 0.54).length;
  const below50 = rows.filter(r => (r.oos_accuracy ?? 0) < 0.50).length;

  return (
    <div className="space-y-3">
      {/* Summary bar */}
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <span className="text-muted-foreground font-medium">{rows.length} stocks</span>
        <span className="flex items-center gap-1.5 text-green-400">
          <span className="w-2.5 h-2.5 rounded bg-green-500/80 inline-block"/>
          {above54} above 54%
        </span>
        <span className="flex items-center gap-1.5 text-yellow-400">
          <span className="w-2.5 h-2.5 rounded bg-yellow-500/30 inline-block"/>
          {above50} at 50–54%
        </span>
        <span className="flex items-center gap-1.5 text-red-400">
          <span className="w-2.5 h-2.5 rounded bg-red-500/20 inline-block"/>
          {below50} below 50%
        </span>
        <span className={cn(
          'text-xs ml-auto px-2 py-0.5 rounded font-mono',
          hasDL ? 'bg-purple-500/10 text-purple-400' : 'bg-yellow-500/10 text-yellow-500'
        )}>
          {hasDL ? '5-model ensemble (LGB+XGB+BiLSTM+TCN-T+NBEATS)' : 'fast mode — LGB+XGB only · run full mode for +3 DL models'}
        </span>
      </div>

      <div className="w-full overflow-x-auto rounded-xl border border-border">
        <table className="w-full text-xs">
          <thead>
            {/* Group header */}
            <tr className="border-b border-border/50 bg-muted/20">
              <th className="sticky left-0 bg-card z-10 px-3 py-1.5" />
              {hasTree && (
                <th colSpan={treeCount} className="text-center px-2 py-1 text-[10px] text-blue-400/70 font-medium tracking-wide">
                  TREE MODELS
                </th>
              )}
              {hasDL && (
                <th colSpan={dlCount} className="text-center px-2 py-1 text-[10px] text-purple-400/70 font-medium tracking-wide">
                  DEEP LEARNING
                </th>
              )}
              <th className="text-center px-2 py-1 text-[10px] text-green-400/70 font-medium tracking-wide">
                ENSEMBLE
              </th>
            </tr>
            {/* Column names */}
            <tr className="border-b border-border bg-muted/30">
              <th className="text-left px-3 py-2 font-medium sticky left-0 bg-card z-10 min-w-[110px]">Symbol</th>
              {cols.map(m => (
                <th key={m.key}
                  className={cn(
                    'px-2 py-2 font-medium text-center min-w-[52px] whitespace-nowrap',
                    m.group === 'ens'  ? 'text-green-400' :
                    m.group === 'tree' ? 'text-blue-400'  : 'text-purple-400'
                  )}>
                  {m.label}
                </th>
              ))}
            </tr>
            {/* Column averages */}
            <tr className="border-b border-border bg-muted/10">
              <td className="px-3 py-1 text-[10px] text-muted-foreground sticky left-0 bg-card z-10 font-medium">
                AVG
              </td>
              {cols.map(m => {
                const avg = colAvg(m.key);
                return (
                  <td key={m.key}
                    className={cn('px-2 py-1 text-center tabular-nums text-[10px]', cellColor(avg, m.group === 'ens'))}>
                    {avg !== null ? (avg * 100).toFixed(1) : '—'}
                  </td>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const ensVal = getVal(row, 'oos_accuracy');
              // Best individual model (exclude ensemble)
              const bestKey = cols
                .filter(m => m.group !== 'ens')
                .reduce((best, m) => {
                  const v = getVal(row, m.key);
                  const b = best ? getVal(row, best) : null;
                  return (v ?? 0) > (b ?? 0) ? m.key : best;
                }, '');

              return (
                <tr key={row.symbol}
                  className={cn('border-b border-border/30 hover:bg-muted/10 transition-colors',
                    i % 2 === 0 ? 'bg-card' : 'bg-muted/5'
                  )}>
                  <td className="px-3 py-1.5 font-mono font-bold sticky left-0 bg-inherit z-10">
                    <div className="flex items-center gap-1.5">
                      <span className={cn(
                        (ensVal ?? 0) >= 0.54 ? 'text-green-400' :
                        (ensVal ?? 0) >= 0.50 ? 'text-foreground' : 'text-muted-foreground'
                      )}>
                        {row.symbol}
                      </span>
                      {row.tradeable && (
                        <span className="text-[9px] bg-green-500/20 text-green-400 px-1 rounded font-normal">live</span>
                      )}
                    </div>
                  </td>
                  {cols.map(m => {
                    const val    = getVal(row, m.key);
                    const isEns  = m.group === 'ens';
                    const isBest = !isEns && hasTree && m.key === bestKey && val !== null;
                    return (
                      <td key={m.key}
                        className={cn(
                          'px-2 py-1.5 text-center tabular-nums rounded transition-colors',
                          cellColor(val, isEns),
                          isBest && 'ring-1 ring-inset ring-white/30',
                          isEns  && 'border-l border-border/40'
                        )}>
                        {val !== null ? (val * 100).toFixed(1) : '—'}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>

        <div className="px-3 py-2.5 border-t border-border flex flex-wrap items-center gap-4 text-xs text-muted-foreground bg-muted/10">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-green-500/85 inline-block"/>≥58%</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-green-500/40 inline-block"/>54–58%</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-yellow-500/25 inline-block"/>50–54%</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-500/20 inline-block"/>&lt;50%</span>
          <span className="ml-auto opacity-60">Ring = best individual model · live = tradeable stock</span>
        </div>
      </div>
    </div>
  );
}

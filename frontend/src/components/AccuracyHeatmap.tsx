'use client';
import { cn } from '@/lib/utils';

const MODELS = [
  { key: 'avg_lgbm_acc',            label: 'LGB',    short: 'L' },
  { key: 'avg_xgb_acc',             label: 'XGB',    short: 'X' },
  { key: 'avg_bilstm_acc',          label: 'BiLSTM', short: 'B' },
  { key: 'avg_tcn_transformer_acc', label: 'TCN',    short: 'T' },
  { key: 'avg_nbeats_acc',          label: 'NBEATS', short: 'N' },
  { key: 'oos_accuracy',            label: 'ENS',    short: 'E' },
];

function cellClass(val: number | null) {
  if (val === null || val === 0) return 'text-muted-foreground/30';
  if (val >= 0.58) return 'bg-green-500/85 text-white font-bold';
  if (val >= 0.54) return 'bg-green-500/40 text-green-200 font-semibold';
  if (val >= 0.52) return 'bg-emerald-500/20 text-emerald-300';
  if (val >= 0.50) return 'bg-yellow-500/25 text-yellow-200';
  return 'bg-red-500/20 text-red-300';
}

function getVal(row: any, key: string): number | null {
  const v = row[key];
  if (v === undefined || v === null || v === 0 || v === '—') return null;
  const n = parseFloat(v);
  return isNaN(n) || n === 0 ? null : n;
}

// Detect if a run has per-model data or just ensemble
function hasPerModelData(stocks: any[]): boolean {
  const sample = stocks.find(s => s.symbol !== 'AVERAGE');
  if (!sample) return false;
  return MODELS.slice(0, 5).some(m => {
    const v = sample[m.key];
    return v !== undefined && v !== null && v !== 0 && v !== '—';
  });
}

export default function AccuracyHeatmap({ stocks }: { stocks: any[] }) {
  const rows      = stocks.filter(s => s.symbol !== 'AVERAGE');
  const hasModels = hasPerModelData(rows);
  const cols      = hasModels ? MODELS : [MODELS[5]]; // Only ENS if no per-model data

  const above54   = rows.filter(r => (r.oos_accuracy ?? 0) >= 0.54).length;
  const above50   = rows.filter(r => (r.oos_accuracy ?? 0) >= 0.50 && (r.oos_accuracy ?? 0) < 0.54).length;
  const below50   = rows.filter(r => (r.oos_accuracy ?? 0) < 0.50).length;

  return (
    <div className="space-y-3">
      {/* Summary bar */}
      <div className="flex items-center gap-6 text-sm">
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
        {!hasModels && (
          <span className="text-xs text-muted-foreground/60 ml-auto">
            Per-model breakdown available in newer runs (fast mode)
          </span>
        )}
      </div>

      <div className="w-full overflow-x-auto rounded-xl border border-border">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border bg-muted/30">
              <th className="text-left px-3 py-2.5 font-medium sticky left-0 bg-card z-10 min-w-[110px]">Symbol</th>
              {cols.map(m => (
                <th key={m.key} className="px-2 py-2.5 font-medium text-center min-w-[60px] whitespace-nowrap">
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const ensVal   = getVal(row, 'oos_accuracy');
              const bestKey  = MODELS.slice(0, 5).reduce((best, m) => {
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
                    <span className={cn(
                      (ensVal ?? 0) >= 0.54 ? 'text-green-400' :
                      (ensVal ?? 0) >= 0.50 ? 'text-foreground' : 'text-muted-foreground'
                    )}>
                      {row.symbol}
                    </span>
                  </td>
                  {cols.map(m => {
                    const val     = getVal(row, m.key);
                    const isBest  = hasModels && m.key !== 'oos_accuracy' && m.key === bestKey;
                    return (
                      <td key={m.key}
                        className={cn(
                          'px-2 py-1.5 text-center tabular-nums rounded transition-colors',
                          cellClass(val),
                          isBest && 'ring-1 ring-inset ring-white/20'
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
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-green-500/85 inline-block"/>≥58% target</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-green-500/40 inline-block"/>54–58%</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-yellow-500/25 inline-block"/>50–54%</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-500/20 inline-block"/>&lt;50%</span>
          {hasModels && <span className="ml-auto opacity-60">Ring = best individual model per stock</span>}
        </div>
      </div>
    </div>
  );
}

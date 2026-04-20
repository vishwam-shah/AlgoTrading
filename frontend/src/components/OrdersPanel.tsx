'use client';
import { useState, useEffect } from 'react';
import { CheckCircle2, XCircle, Clock, Loader2, Play, AlertTriangle, Activity, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

interface Order {
  symbol: string; direction: string; prob_up?: number;
  target_pct?: number; order_value?: number; qty?: number; price?: number;
}

interface ExecutionFill {
  symbol: string; side: string; filled_qty: number;
  avg_price: number; order_value: number; status: string;
  net_cost: number; placed_at: string;
}

interface OrdersPanelProps {
  orders: Order[];
  orderCount?: number;       // total from API (may be > orders.length due to slicing)
  totalDeployed?: number;    // total deployed INR from API
  runId?: string;
}

export default function OrdersPanel({ orders, orderCount, totalDeployed: totalDeployedProp, runId }: OrdersPanelProps) {
  const [executing,   setExecuting]  = useState(false);
  const [jobId,       setJobId]      = useState<string | null>(null);
  const [jobStatus,   setJobStatus]  = useState<string>('');
  const [fills,       setFills]      = useState<ExecutionFill[]>([]);
  const [paperMode,   setPaperMode]  = useState(true);
  const [showAllFills, setShowAllFills] = useState(false);

  // Load existing fills on mount so already-executed orders show correct status
  useEffect(() => {
    fetch('/api/v3/execution/history?limit=200')
      .then(r => r.json())
      .then(d => { if (d?.trades?.length) setFills(d.trades); })
      .catch(() => {});
  }, []);

  // Map symbol → most recent fill for O(1) lookup
  const fillMap = new Map<string, ExecutionFill>();
  for (const f of fills) {
    if (!fillMap.has(f.symbol) || f.placed_at > (fillMap.get(f.symbol)?.placed_at ?? '')) {
      fillMap.set(f.symbol, f);
    }
  }

  async function executeOrders() {
    setExecuting(true);
    setFills([]);
    try {
      const res = await fetch(`/api/v3/angel/execute-today?paper=${paperMode}`, { method: 'POST' });
      if (!res.ok) { toast.error('Execution failed'); return; }
      const data = await res.json();
      setJobId(data.job_id);
      setJobStatus('running');
      toast.info(`${paperMode ? 'Paper' : 'Live'} execution started`);

      // Poll for completion
      const poll = setInterval(async () => {
        const ctrl2 = new AbortController();
        const tid2  = setTimeout(() => ctrl2.abort(), 5000);
        const status = await fetch(`/api/v1/v3/${data.job_id}/status`, { signal: ctrl2.signal })
          .then(r => r.json()).catch(() => null).finally(() => clearTimeout(tid2));
        if (status?.status === 'completed') {
          clearInterval(poll);
          setJobStatus('completed');
          toast.success('Execution complete');
          setExecuting(false);
          // Refresh fills so status column updates
          fetch('/api/v3/execution/history?limit=200')
            .then(r => r.json())
            .then(d => { if (d?.trades?.length) setFills(d.trades); })
            .catch(() => {});
        } else if (status?.status === 'failed') {
          clearInterval(poll);
          setJobStatus('failed');
          toast.error(status.message);
          setExecuting(false);
        }
      }, 3000);
    } catch { toast.error('Backend unreachable'); setExecuting(false); }
  }

  // Use API totals when available — orders array may be sliced to 10 items
  const totalDeployed = totalDeployedProp ?? orders.reduce((s, o) => s + (o.order_value ?? 0), 0);
  const displayCount  = orderCount ?? orders.length;
  // Orders from signal_publisher use direction='BUY'; predictions use 'UP'
  const buyOrders     = orders.filter(o => ['BUY','UP'].includes(o.direction) || !o.direction);

  return (
    <div className="space-y-4">
      {/* Execute bar */}
      <div className="flex items-center gap-3 p-4 rounded-xl border border-border bg-card">
        <div className="flex-1">
          <div className="text-sm font-medium">
            {displayCount} approved orders · ₹{totalDeployed.toLocaleString('en-IN', { maximumFractionDigits: 0 })} total
          </div>
          <div className="text-xs text-muted-foreground mt-0.5">
            {runId ? `Run: ${runId}` : 'No run selected'} · Kelly-sized, risk-validated
          </div>
        </div>

        {/* Paper / Live toggle */}
        <div className="flex items-center gap-2 text-xs">
          <span className={cn('font-medium', paperMode ? 'text-yellow-400' : 'text-muted-foreground')}>Paper</span>
          <button
            onClick={() => setPaperMode(p => !p)}
            className={cn(
              'relative w-10 h-5 rounded-full transition-colors',
              paperMode ? 'bg-yellow-500/30 border border-yellow-500/40' : 'bg-red-500/30 border border-red-500/40'
            )}
          >
            <span className={cn('absolute top-0.5 w-4 h-4 rounded-full transition-all',
              paperMode ? 'left-0.5 bg-yellow-400' : 'left-5 bg-red-400'
            )} />
          </button>
          <span className={cn('font-medium', !paperMode ? 'text-red-400' : 'text-muted-foreground')}>Live</span>
        </div>

        {!paperMode && (
          <div className="flex items-center gap-1 text-xs text-red-400 bg-red-500/10 px-2 py-1 rounded-lg border border-red-500/20">
            <AlertTriangle className="h-3 w-3" /> Real money
          </div>
        )}

        <button
          onClick={executeOrders}
          disabled={executing || !orders.length}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all disabled:opacity-50',
            paperMode
              ? 'bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-400 border border-yellow-500/30'
              : 'bg-red-500 hover:bg-red-400 text-white shadow-lg shadow-red-500/20'
          )}
        >
          {executing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {executing ? 'Executing…' : `Execute ${paperMode ? '(Paper)' : '(Live)'}`}
        </button>
      </div>

      {/* Job status */}
      {jobId && (
        <div className={cn(
          'flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm border',
          jobStatus === 'completed' ? 'bg-green-500/10 border-green-500/20 text-green-400'
          : jobStatus === 'failed'  ? 'bg-red-500/10 border-red-500/20 text-red-400'
          : 'bg-muted/20 border-border text-muted-foreground'
        )}>
          {jobStatus === 'completed' ? <CheckCircle2 className="h-4 w-4" />
          : jobStatus === 'failed'   ? <XCircle className="h-4 w-4" />
          : <Clock className="h-4 w-4 animate-pulse" />}
          Job {jobId} · {jobStatus}
        </div>
      )}

      {/* Orders table */}
      {orders.length > 0 ? (
        <div className="rounded-xl border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-muted/20 border-b border-border">
              <tr>
                {['Symbol','P(UP)','Kelly %','Amount','Qty','Price','Status'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-xs font-medium text-muted-foreground uppercase tracking-wide">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {orders.map((o, i) => (
                <tr key={i} className="hover:bg-muted/10 transition-colors">
                  <td className="px-4 py-3 font-mono font-bold">{o.symbol}</td>
                  <td className="px-4 py-3 font-mono">
                    <span className={cn('font-bold', (o.prob_up ?? 0) >= 0.58 ? 'text-green-400' : (o.prob_up ?? 0) >= 0.52 ? 'text-yellow-400' : 'text-muted-foreground')}>
                      {o.prob_up ? `${(o.prob_up * 100).toFixed(1)}%` : '—'}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono text-indigo-400">{o.target_pct ?? '—'}%</td>
                  <td className="px-4 py-3 font-mono">₹{o.order_value?.toLocaleString('en-IN', { maximumFractionDigits: 0 }) ?? '—'}</td>
                  <td className="px-4 py-3 font-mono">{o.qty ?? '—'}</td>
                  <td className="px-4 py-3 font-mono text-muted-foreground">₹{o.price?.toFixed(2) ?? '—'}</td>
                  <td className="px-4 py-3">
                    {(() => {
                      const fill = fillMap.get(o.symbol);
                      if (!fill) return (
                        <span className="flex items-center gap-1 text-xs text-yellow-400 bg-yellow-500/10 px-2 py-0.5 rounded-full border border-yellow-500/20">
                          <Clock className="h-2.5 w-2.5" /> Pending
                        </span>
                      );
                      const isFilled = fill.status === 'FILLED';
                      return (
                        <span className={cn('flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border',
                          isFilled
                            ? 'text-green-400 bg-green-500/10 border-green-500/20'
                            : 'text-red-400 bg-red-500/10 border-red-500/20'
                        )}>
                          {isFilled ? <CheckCircle2 className="h-2.5 w-2.5" /> : <XCircle className="h-2.5 w-2.5" />}
                          {fill.status}
                        </span>
                      );
                    })()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-10 text-muted-foreground gap-2 rounded-xl border border-border">
          <CheckCircle2 className="h-7 w-7 opacity-30" />
          <p className="text-sm">No orders yet</p>
          <p className="text-xs opacity-60">Run signal_publisher after pipeline completes</p>
        </div>
      )}

      {/* Recent Fills */}
      {fills.length > 0 && (
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold">Recent Fills</span>
              <span className="text-xs text-muted-foreground">({fills.length} total)</span>
            </div>
            {fills.length > 5 && (
              <button onClick={() => setShowAllFills(v => !v)}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
                {showAllFills ? 'Show less' : `Show all ${fills.length}`}
                <ChevronDown className={cn('h-3 w-3 transition-transform', showAllFills && 'rotate-180')} />
              </button>
            )}
          </div>
          <table className="w-full text-xs">
            <thead className="bg-muted/10 text-muted-foreground border-b border-border">
              <tr>{['Symbol','Side','Qty','Avg Price','Value','Charges','Status','Date'].map(h =>
                <th key={h} className="text-left px-4 py-2 font-medium whitespace-nowrap">{h}</th>
              )}</tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {fills.slice(0, showAllFills ? fills.length : 5).map((t, i) => (
                <tr key={i} className="hover:bg-muted/10 transition-colors">
                  <td className="px-4 py-2.5 font-mono font-bold">{t.symbol}</td>
                  <td className="px-4 py-2.5">
                    <span className={cn('text-xs font-semibold px-2 py-0.5 rounded-full',
                      t.side === 'BUY' ? 'bg-green-500/15 text-green-400' : 'bg-red-500/15 text-red-400'
                    )}>{t.side}</span>
                  </td>
                  <td className="px-4 py-2.5 font-mono">{t.filled_qty}</td>
                  <td className="px-4 py-2.5 font-mono text-muted-foreground">₹{(t.avg_price ?? 0).toFixed(2)}</td>
                  <td className="px-4 py-2.5 font-mono">₹{(t.order_value ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}</td>
                  <td className="px-4 py-2.5 font-mono text-red-400">
                    ₹{((t.net_cost ?? 0) - (t.order_value ?? 0)).toFixed(0)}
                  </td>
                  <td className="px-4 py-2.5">
                    <span className={cn('text-xs px-2 py-0.5 rounded-full border',
                      t.status === 'FILLED' ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-muted text-muted-foreground border-border'
                    )}>{t.status}</span>
                  </td>
                  <td className="px-4 py-2.5 text-muted-foreground">
                    {t.placed_at ? new Date(t.placed_at).toLocaleDateString('en-IN') : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

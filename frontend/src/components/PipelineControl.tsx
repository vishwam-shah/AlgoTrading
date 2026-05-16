'use client';
import { useState, useCallback, useEffect } from 'react';
import { Play, Square, RefreshCw, ChevronDown, ChevronUp, Settings2, Loader2, CheckCircle2, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';


interface Config {
  fastMode: boolean;
  forceFeatures: boolean;
  capital: number;
  minConfidence: number;
  holdDays: number;
  stopLossPct: number;
  takeProfitPct: number;
}

const DEFAULT_CONFIG: Config = {
  fastMode: true, forceFeatures: false, capital: 500000,
  minConfidence: 0.52, holdDays: 7, stopLossPct: 3.0, takeProfitPct: 6.0,
};

interface PipelineControlProps {
  onComplete?: () => void;
}

export default function PipelineControl({ onComplete }: PipelineControlProps) {
  const [open,         setOpen]         = useState(false);
  const [allSymbols,   setAllSymbols]   = useState<string[]>([]);
  const [sectors,      setSectors]      = useState<Record<string, string[]>>({});
  const [universeLoaded, setUniverseLoaded] = useState(false);
  const [selected,     setSelected]     = useState<Set<string>>(new Set());
  const [config,       setConfig]       = useState<Config>(DEFAULT_CONFIG);
  const [running,      setRunning]       = useState(false);
  const [jobId,        setJobId]        = useState<string | null>(null);
  const [jobStatus,    setJobStatus]    = useState<'idle'|'running'|'completed'|'failed'>('idle');
  const [progress,     setProgress]     = useState(0);
  const [statusMsg,    setStatusMsg]    = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [searchSym,    setSearchSym]    = useState('');

  // Fetch the symbol universe + sector map from the backend on mount.
  // Source of truth: V3/00_config/config.py — never hardcode here.
  useEffect(() => {
    let cancelled = false;
    fetch('/api/v3/symbols')
      .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
      .then(data => {
        if (cancelled) return;
        const syms: string[]                  = data.symbols ?? [];
        const secs: Record<string, string[]>  = data.sectors ?? {};
        setAllSymbols(syms);
        setSectors(secs);
        setSelected(new Set(syms));   // default: all selected
        setUniverseLoaded(true);
      })
      .catch(err => {
        if (cancelled) return;
        toast.error(`Failed to load stock universe: ${err.message}`);
        setUniverseLoaded(true);
      });
    return () => { cancelled = true; };
  }, []);

  const toggleSym   = (s: string) => setSelected(prev => { const n = new Set(prev); n.has(s) ? n.delete(s) : n.add(s); return n; });
  const selectAll   = () => setSelected(new Set(allSymbols));
  const selectNone  = () => setSelected(new Set());
  const selectSector = (syms: string[]) => setSelected(prev => {
    const n = new Set(prev);
    const allIn = syms.every(s => n.has(s));
    syms.forEach(s => allIn ? n.delete(s) : n.add(s));
    return n;
  });

  const filtered = searchSym
    ? allSymbols.filter(s => s.toLowerCase().includes(searchSym.toLowerCase()))
    : allSymbols;

  async function startPipeline() {
    if (selected.size === 0) { toast.error('Select at least one stock'); return; }
    setRunning(true); setJobStatus('running'); setProgress(0);
    setStatusMsg('Starting pipeline…');

    const symbols = Array.from(selected);
    const params  = new URLSearchParams({
      fast: String(config.fastMode),
      force_features: String(config.forceFeatures),
    });
    symbols.forEach(s => params.append('symbols', s));

    try {
      const res = await fetch(`/api/v1/v3/run?${params}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols, capital: config.capital }),
      });
      if (!res.ok) { throw new Error(await res.text()); }
      const data = await res.json();
      setJobId(data.job_id);
      toast.success(`Pipeline started — ${symbols.length} stocks`);

      // Poll status
      const poll = setInterval(async () => {
        const ctrl = new AbortController();
        const tid  = setTimeout(() => ctrl.abort(), 5000);
        const s = await fetch(`/api/v1/v3/${data.job_id}/status`, { signal: ctrl.signal })
          .then(r => r.json()).catch(() => null).finally(() => clearTimeout(tid));
        if (!s) return;
        setProgress(s.progress ?? 0);
        setStatusMsg(s.message ?? '');
        if (s.status === 'completed') {
          clearInterval(poll); setJobStatus('completed'); setRunning(false);
          toast.success('Pipeline complete! Dashboard refreshing…');
          onComplete?.();
        } else if (s.status === 'failed') {
          clearInterval(poll); setJobStatus('failed'); setRunning(false);
          toast.error(`Pipeline failed: ${s.message}`);
        }
      }, 5000);
    } catch (e: any) {
      toast.error(e.message); setRunning(false); setJobStatus('failed');
    }
  }

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-5 py-4 hover:bg-muted/10 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-lg bg-indigo-500/20 flex items-center justify-center">
            <Settings2 className="h-4 w-4 text-indigo-400" />
          </div>
          <div className="text-left">
            <div className="text-sm font-semibold">Pipeline Control</div>
            <div className="text-xs text-muted-foreground">
              {selected.size} of {allSymbols.length} stocks selected · {config.fastMode ? 'Fast (LGB+XGB)' : 'Full (5 models)'} · ₹{(config.capital/100000).toFixed(1)}L capital
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {jobStatus === 'running' && (
            <div className="flex items-center gap-2 text-xs text-indigo-400">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              {progress}% — {statusMsg.slice(0, 40)}
            </div>
          )}
          {jobStatus === 'completed' && <CheckCircle2 className="h-4 w-4 text-green-400" />}
          {jobStatus === 'failed'    && <XCircle      className="h-4 w-4 text-red-400" />}
          {open ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
        </div>
      </button>

      {/* Progress bar when running */}
      {jobStatus === 'running' && (
        <div className="h-0.5 bg-muted/30">
          <div className="h-0.5 bg-indigo-500 transition-all duration-1000" style={{ width: `${progress}%` }} />
        </div>
      )}

      {open && (
        <div className="border-t border-border p-5 space-y-5">

          {/* Sector quick-select */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Select Stocks</span>
              <div className="flex gap-2">
                <button onClick={selectAll}  className="text-xs px-2 py-1 rounded-md bg-indigo-500/10 text-indigo-400 hover:bg-indigo-500/20">All</button>
                <button onClick={selectNone} className="text-xs px-2 py-1 rounded-md bg-muted/30 text-muted-foreground hover:bg-muted/50">None</button>
              </div>
            </div>

            {/* Sector chips */}
            <div className="flex flex-wrap gap-2 mb-3">
              {Object.entries(sectors).map(([sector, syms]) => {
                const allIn = syms.every(s => selected.has(s));
                return (
                  <button key={sector} onClick={() => selectSector(syms)}
                    className={cn('px-2.5 py-1 rounded-lg text-xs border transition-all',
                      allIn ? 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30'
                             : 'border-border text-muted-foreground hover:bg-muted/30'
                    )}>
                    {sector} ({syms.length})
                  </button>
                );
              })}
            </div>

            {/* Search + stock grid */}
            <input
              value={searchSym}
              onChange={e => setSearchSym(e.target.value)}
              placeholder="Search symbol…"
              className="w-full mb-2 px-3 py-1.5 rounded-lg border border-border bg-muted/20 text-xs outline-none focus:border-primary/50"
            />
            <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto">
              {filtered.map(s => (
                <button key={s} onClick={() => toggleSym(s)}
                  className={cn('px-2 py-0.5 rounded font-mono text-xs border transition-all',
                    selected.has(s)
                      ? 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30'
                      : 'border-border text-muted-foreground hover:border-border/80'
                  )}>
                  {s}
                </button>
              ))}
            </div>
            <div className="text-[10px] text-muted-foreground mt-1">
              {universeLoaded ? `${selected.size} / ${allSymbols.length} selected` : 'Loading universe…'}
            </div>
          </div>

          {/* Config */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {/* Fast mode toggle */}
            <div className="rounded-lg border border-border p-3">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wide mb-2">Model Mode</div>
              <div className="flex gap-2">
                {[{ v: true, l: 'Fast' }, { v: false, l: 'Full' }].map(o => (
                  <button key={String(o.v)} onClick={() => setConfig(c => ({ ...c, fastMode: o.v }))}
                    className={cn('flex-1 py-1.5 rounded-md text-xs font-medium border transition-all',
                      config.fastMode === o.v
                        ? 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30'
                        : 'border-border text-muted-foreground hover:bg-muted/30'
                    )}>{o.l}</button>
                ))}
              </div>
              <div className="text-[9px] text-muted-foreground mt-1">
                {config.fastMode ? 'LGB+XGB ~30min' : '5 models ~4hrs'}
              </div>
            </div>

            {/* Capital */}
            <div className="rounded-lg border border-border p-3">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Capital (₹)</div>
              <input type="number" value={config.capital} onChange={e => setConfig(c => ({ ...c, capital: parseInt(e.target.value)||500000 }))}
                className="w-full bg-transparent text-sm font-mono font-bold outline-none" />
              <div className="flex gap-1 mt-1">
                {[100000,500000,1000000].map(v => (
                  <button key={v} onClick={() => setConfig(c => ({ ...c, capital: v }))}
                    className="text-[9px] px-1.5 py-0.5 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground">
                    ₹{v>=100000?`${v/100000}L`:`${v/1000}K`}
                  </button>
                ))}
              </div>
            </div>

            {/* Min confidence */}
            <div className="rounded-lg border border-border p-3">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Min P(UP)</div>
              <input type="number" step="0.01" min="0.50" max="0.90" value={config.minConfidence}
                onChange={e => setConfig(c => ({ ...c, minConfidence: parseFloat(e.target.value)||0.52 }))}
                className="w-full bg-transparent text-sm font-mono font-bold outline-none" />
              <div className="text-[9px] text-muted-foreground mt-1">Filter signal threshold</div>
            </div>

            {/* Hold days */}
            <div className="rounded-lg border border-border p-3">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Hold Days</div>
              <input type="number" min="1" max="30" value={config.holdDays}
                onChange={e => setConfig(c => ({ ...c, holdDays: parseInt(e.target.value)||7 }))}
                className="w-full bg-transparent text-sm font-mono font-bold outline-none" />
              <div className="text-[9px] text-muted-foreground mt-1">7d = break-even threshold</div>
            </div>

            {/* Stop loss */}
            <div className="rounded-lg border border-border p-3">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Stop Loss %</div>
              <input type="number" step="0.5" min="1" max="10" value={config.stopLossPct}
                onChange={e => setConfig(c => ({ ...c, stopLossPct: parseFloat(e.target.value)||3 }))}
                className="w-full bg-transparent text-sm font-mono font-bold outline-none" />
              <div className="text-[9px] text-muted-foreground mt-1">Exit if down this %</div>
            </div>

            {/* Take profit */}
            <div className="rounded-lg border border-border p-3">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Take Profit %</div>
              <input type="number" step="0.5" min="1" max="30" value={config.takeProfitPct}
                onChange={e => setConfig(c => ({ ...c, takeProfitPct: parseFloat(e.target.value)||6 }))}
                className="w-full bg-transparent text-sm font-mono font-bold outline-none" />
              <div className="text-[9px] text-muted-foreground mt-1">Lock gains target</div>
            </div>

            {/* Force recompute */}
            <div className="rounded-lg border border-border p-3">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wide mb-2">Force Features</div>
              <button onClick={() => setConfig(c => ({ ...c, forceFeatures: !c.forceFeatures }))}
                className={cn('w-full py-1.5 rounded-md text-xs font-medium border transition-all',
                  config.forceFeatures
                    ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                    : 'border-border text-muted-foreground hover:bg-muted/30'
                )}>
                {config.forceFeatures ? 'ON — recompute' : 'OFF — use cache'}
              </button>
            </div>
          </div>

          {/* Run button */}
          <div className="flex items-center gap-3 pt-1">
            <button
              onClick={startPipeline}
              disabled={running || selected.size === 0}
              className={cn(
                'flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-semibold transition-all disabled:opacity-50',
                'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 hover:from-indigo-400 hover:to-purple-500'
              )}
            >
              {running
                ? <><Loader2 className="h-4 w-4 animate-spin" /> Running ({progress}%)…</>
                : <><Play className="h-4 w-4" /> Run Pipeline ({selected.size} stocks)</>}
            </button>

            <button onClick={() => setConfig(DEFAULT_CONFIG)}
              className="flex items-center gap-1.5 px-3 py-2.5 rounded-xl text-xs border border-border hover:bg-muted/30 transition-colors text-muted-foreground">
              <RefreshCw className="h-3.5 w-3.5" /> Reset
            </button>

            <div className="ml-auto text-xs text-muted-foreground">
              Est: {config.fastMode ? `~${Math.ceil(selected.size * 0.35)} min` : `~${Math.ceil(selected.size * 2.5)} min`}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

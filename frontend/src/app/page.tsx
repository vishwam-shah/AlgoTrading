'use client';
import { useState, useEffect, useCallback, useMemo } from 'react';
import { useTheme } from 'next-themes';
import dynamic from 'next/dynamic';
import {
  Brain, Sun, Moon, TrendingUp, TrendingDown, Wallet,
  BarChart2, Activity, RefreshCw, CheckCircle2,
  LineChart, ShoppingCart, PieChart, Award,
  ChevronDown, History, Zap, FlaskConical,
  Download,
} from 'lucide-react';
import { Toaster } from 'sonner';
import { cn } from '@/lib/utils';
import { apiFetch, useLivePrices } from '@/hooks/useData';
import StatCard from '@/components/StatCard';
import SignalsTable from '@/components/SignalsTable';
import AccuracyHeatmap from '@/components/AccuracyHeatmap';
import StockSearch from '@/components/StockSearch';
import PortfolioPanel from '@/components/PortfolioPanel';
import ExecutionPanel from '@/components/ExecutionPanel';
import OrdersPanel from '@/components/OrdersPanel';
import OOSMetricsPanel from '@/components/OOSMetricsPanel';
import BacktestPanel from '@/components/BacktestPanel';
import PaperTradingPanel from '@/components/PaperTradingPanel';
import ResearchChecklist from '@/components/ResearchChecklist';
import SentimentPanel from '@/components/SentimentPanel';
import RunContextBanner from '@/components/RunContextBanner';
import WalletWidget from '@/components/WalletWidget';
import PipelineControl from '@/components/PipelineControl';
import type { BacktestResults } from '@/components/BacktestPanel';

const CandlestickChart = dynamic(() => import('@/components/CandlestickChart'), { ssr: false });
const LivePriceTicker  = dynamic(() => import('@/components/LivePriceTicker'),  { ssr: false });

// ── Types ────────────────────────────────────────────────────────────────────

interface Dashboard {
  run_id?: string;
  pipeline?: { n_stocks: number; n_total: number; is_complete: boolean; avg_accuracy: number };
  predictions?: { total: number; up: number; down: number };
  orders?: { count: number; total_deployed: number; orders: any[] };
  sentiment?: { n_symbols: number; n_dates: number; latest_date: string };
  trade_history?: { total_trades: number };
}
interface RunSummary  { stocks: any[] }
interface Prediction  {
  symbol: string;
  direction: 'UP'|'DOWN';
  prob_up?: number;
  confidence?: number;
  prediction_date?: string;
  prediction_for?: string;
  last_close?: number;
  predicted_price?: number;
  predicted_move_pct?: number;
  range_low?: number;
  range_high?: number;
  range_down_pct?: number;
  range_up_pct?: number;
  ensemble_accuracy?: number;
  directional_accuracy_for_signal?: number;
  up_signal_accuracy?: number;
  down_signal_accuracy?: number;
  best_model?: string;
  best_model_accuracy?: number;
}
interface AngelStatus { credentials_present: boolean; login_ok?: boolean; message?: string }
interface AngelFunds  { funds: { available: number; net: number; used_margin: number } }
interface VirtualWallet {
  balance: number; initial_balance: number; portfolio_value: number;
  total_value: number; total_pnl: number; total_pnl_pct: number;
  realized_pnl: number; unrealized_pnl: number; holdings_count: number;
  portfolio: Record<string, any>;
}
interface Holdings { holdings: any[]; total_pnl: number; count: number }

// ── Nav structure ────────────────────────────────────────────────────────────

type ViewMode  = 'trade' | 'research';
type TradeTab  = 'overview' | 'signals' | 'orders' | 'portfolio' | 'chart';
type ResearchTab = 'accuracy' | 'analysis' | 'runs';

const TRADE_TABS: { id: TradeTab; label: string; icon: any }[] = [
  { id: 'overview',  label: 'Overview',  icon: Zap         },
  { id: 'signals',   label: 'Signals',   icon: TrendingUp  },
  { id: 'orders',    label: 'Orders',    icon: ShoppingCart },
  { id: 'portfolio', label: 'Portfolio', icon: PieChart    },
  { id: 'chart',     label: 'Chart',     icon: LineChart   },
];
const RESEARCH_TABS: { id: ResearchTab; label: string; icon: any }[] = [
  { id: 'accuracy', label: 'Accuracy', icon: BarChart2   },
  { id: 'analysis', label: 'Analysis', icon: Award       },
  { id: 'runs',     label: 'Runs',     icon: History     },
];

// ── Component ────────────────────────────────────────────────────────────────

export default function TradingDashboard() {
  const { theme, setTheme } = useTheme();
  const [mounted,       setMounted]       = useState(false);
  const [viewMode,      setViewMode]      = useState<ViewMode>('trade');
  const [tradeTab,      setTradeTab]      = useState<TradeTab>('overview');
  const [researchTab,   setResearchTab]   = useState<ResearchTab>('accuracy');
  const [dashboard,     setDashboard]     = useState<Dashboard | null>(null);
  const [summary,       setSummary]       = useState<RunSummary | null>(null);
  const [predictions,   setPredictions]   = useState<Prediction[]>([]);
  const [angelStatus,   setAngelStatus]   = useState<AngelStatus | null>(null);
  const [funds,         setFunds]         = useState<AngelFunds | null>(null);
  const [virtualWallet, setVirtualWallet] = useState<VirtualWallet | null>(null);
  const [holdings,      setHoldings]      = useState<Holdings | null>(null);
  const [ohlcv,         setOhlcv]         = useState<any[]>([]);
  const [chartSym,      setChartSym]      = useState('SBIN');
  const [watchlist,     setWatchlist]     = useState<string[]>(['SBIN','HDFCBANK','TCS','INFY','RELIANCE','AXISBANK','KOTAKBANK','WIPRO']);
  const [refreshing,    setRefreshing]    = useState(false);
  const [config] = useState({ capital: 500000 });

  // Shared state for Research/Analysis components → ResearchChecklist
  const [sharedBacktest,     setSharedBacktest]     = useState<BacktestResults | null>(null);
  const [sharedPaperSession, setSharedPaperSession] = useState<any>(null);

  // Run selector
  const [allRuns,       setAllRuns]       = useState<{run_id: string; n_stocks?: number; avg_accuracy?: number}[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  useEffect(() => { setMounted(true); }, []);

  const refreshRunList = useCallback(async () => {
    const r = await apiFetch<{runs: any[]}>('/api/v3/runs');
    if (r?.runs) setAllRuns(r.runs);
  }, []);

  const allSymbols = useMemo(() => {
    const fromSummary = (summary?.stocks || []).map((s: any) => s.symbol).filter((s: string) => s !== 'AVERAGE');
    return Array.from(new Set([...watchlist, ...fromSummary])).slice(0, 50);
  }, [summary, watchlist]);

  const { prices, change, connected } = useLivePrices(allSymbols);

  const refresh = useCallback(async (runId?: string | null) => {
    setRefreshing(true);
    const targetRun = runId !== undefined ? runId : selectedRunId;
    const dashUrl = targetRun ? `/api/v3/dashboard?run_id=${targetRun}` : '/api/v3/dashboard';
    const predUrl = targetRun ? `/api/v3/predictions/${targetRun}` : '/api/v3/predictions/latest';

    const [db, pred, status] = await Promise.all([
      apiFetch<Dashboard>(dashUrl),
      apiFetch<{ predictions: Prediction[] }>(predUrl),
      apiFetch<AngelStatus>('/api/v3/angel/status'),
    ]);
    if (db) {
      setDashboard(db);
      const effectiveRunId = targetRun ?? db.run_id;
      if (effectiveRunId) {
        const sum = await apiFetch<RunSummary>(`/api/v3/runs/${effectiveRunId}/summary`);
        if (sum) setSummary(sum);
      }
    }
    if (pred?.predictions) setPredictions(pred.predictions);
    if (status) setAngelStatus(status);
    setRefreshing(false);
  }, [selectedRunId]);

  const selectRun = useCallback((runId: string | null) => {
    setSelectedRunId(runId);
    refresh(runId);
  }, [refresh]);

  const onPipelineComplete = useCallback(async () => {
    setSelectedRunId(null);
    await refreshRunList();
    await refresh(null);
  }, [refresh, refreshRunList]);

  useEffect(() => {
    refresh();
    refreshRunList();
    const t = setInterval(() => refresh(), 60_000);
    return () => clearInterval(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    apiFetch<VirtualWallet>('/api/v1/wallet').then(w => { if (w) setVirtualWallet(w); });
  }, []);

  useEffect(() => {
    if (!angelStatus?.credentials_present) return;
    Promise.all([apiFetch<AngelFunds>('/api/v3/angel/funds'), apiFetch<Holdings>('/api/v3/angel/holdings')])
      .then(([f, h]) => { if (f) setFunds(f); if (h) setHoldings(h); });
  }, [angelStatus?.credentials_present]);

  useEffect(() => {
    apiFetch<{ history: any[] }>(`/api/v1/market/history/${chartSym}?days=365`)
      .then(r => { if (r?.history) setOhlcv(r.history); });
  }, [chartSym]);

  const toggleWatchlist = useCallback((sym: string) => {
    setWatchlist(w => w.includes(sym) ? w.filter(s => s !== sym) : [...w, sym]);
  }, []);

  // Derived values
  const avgAcc     = dashboard?.pipeline?.avg_accuracy ?? 0;
  const nStocks    = dashboard?.pipeline?.n_stocks ?? 0;
  const nTotal     = dashboard?.pipeline?.n_total ?? 0;
  const isComplete = dashboard?.pipeline?.is_complete ?? false;
  const upCount    = dashboard?.predictions?.up ?? 0;
  const deployed   = dashboard?.orders?.total_deployed ?? 0;
  const angelConnected = (angelStatus?.credentials_present && angelStatus?.login_ok) ?? false;
  const cash     = angelConnected ? (funds?.funds?.available ?? 0) : (virtualWallet?.balance ?? 0);
  const invested = angelConnected
    ? (holdings?.holdings?.reduce((s: number, h: any) => s + (h.qty * (prices[h.symbol] || h.ltp || h.avg_price)), 0) ?? 0)
    : (virtualWallet?.portfolio_value ?? 0);
  const totalPnl   = angelConnected
    ? (typeof holdings?.total_pnl === 'number' ? holdings.total_pnl : 0)
    : (virtualWallet?.unrealized_pnl ?? 0);

  const enrichedStocks = useMemo(() => {
    if (!summary?.stocks) return [];
    const predMap: Record<string, Prediction> = {};
    predictions.forEach(p => { predMap[p.symbol] = p; });
    return summary.stocks
      .filter((s: any) => s.symbol !== 'AVERAGE')
      .map((s: any) => ({
        ...s,
        ...predMap[s.symbol],
        ltp:       prices[s.symbol],
        chg:       change(s.symbol),
      }));
  }, [summary, predictions, prices, change]);

  const chartSignals = predictions
    .filter(p => p.symbol === chartSym && p.direction === 'UP')
    .map(p => ({ date: ohlcv[ohlcv.length - 1]?.date ?? '', direction: 'UP' as const, prob_up: p.prob_up }));

  const summaryStocks = (summary?.stocks ?? []).filter((s: any) => s.symbol !== 'AVERAGE');

  const exportPredictionsCsv = useCallback(() => {
    if (!predictions.length) return;
    const runLabel = selectedRunId ?? dashboard?.run_id ?? 'latest';
    const headers = [
      'symbol',
      'prediction_date',
      'prediction_for',
      'direction',
      'action_probability_up',
      'confidence',
      'last_close',
      'predicted_price',
      'predicted_move_pct',
      'range_low',
      'range_high',
      'range_down_pct',
      'range_up_pct',
      'ensemble_accuracy',
      'directional_accuracy_for_signal',
      'up_signal_accuracy',
      'down_signal_accuracy',
      'best_model',
      'best_model_accuracy',
    ];
    const rows = predictions.map((p) => [
      p.symbol,
      p.prediction_date ?? '',
      p.prediction_for ?? '',
      p.direction,
      p.prob_up ?? '',
      p.confidence ?? '',
      p.last_close ?? '',
      p.predicted_price ?? '',
      p.predicted_move_pct ?? '',
      p.range_low ?? '',
      p.range_high ?? '',
      p.range_down_pct ?? '',
      p.range_up_pct ?? '',
      p.ensemble_accuracy ?? '',
      p.directional_accuracy_for_signal ?? '',
      p.up_signal_accuracy ?? '',
      p.down_signal_accuracy ?? '',
      p.best_model ?? '',
      p.best_model_accuracy ?? '',
    ]);
    const csv = [
      headers.join(','),
      ...rows.map((row) => row.map((value) => `"${String(value).replace(/"/g, '""')}"`).join(',')),
    ].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `signals_${runLabel}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [dashboard?.run_id, predictions, selectedRunId]);

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <Toaster position="top-right" richColors />

      {/* ── Header ── */}
      <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur flex-shrink-0">
        <div className="page-container h-14 flex items-center gap-4">
          <div className="flex items-center gap-2.5 flex-shrink-0">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow shadow-indigo-500/30">
              <Brain className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="text-sm font-bold leading-tight">AlgoTrading V3</div>
              <div className="text-[9px] text-muted-foreground leading-tight">NSE · Walk-Forward Ensemble</div>
            </div>
          </div>

          <div className="flex-1 max-w-lg">
            <StockSearch
              allStocks={enrichedStocks}
              onSelect={sym => { setChartSym(sym); setViewMode('trade'); setTradeTab('chart'); }}
              watchlist={watchlist}
              onWatchlistToggle={toggleWatchlist}
            />
          </div>

          <div className="flex-1" />

          <div className="flex items-center gap-2 flex-shrink-0">
            {/* Angel status + refresh combined */}
            <button
              onClick={() => refresh()}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors',
                angelConnected
                  ? 'bg-green-500/10 text-green-400 border-green-500/20 hover:bg-green-500/15'
                  : angelStatus?.credentials_present
                    ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20 hover:bg-yellow-500/15'
                    : 'bg-muted text-muted-foreground border-border hover:bg-muted/70'
              )}
              title={angelStatus?.message ?? ''}>
              <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0',
                angelConnected ? 'bg-green-400 animate-pulse'
                : angelStatus?.credentials_present ? 'bg-yellow-400'
                : 'bg-muted-foreground'
              )} />
              {angelConnected ? 'Angel One' : angelStatus?.credentials_present ? 'TOTP Invalid' : 'Not configured'}
              <RefreshCw className={cn('h-3 w-3 ml-0.5', refreshing && 'animate-spin')} />
            </button>

            <WalletWidget
              cash={cash}
              invested={invested}
              totalPnl={totalPnl}
              holdings={angelConnected ? (holdings?.holdings?.length ?? 0) : (virtualWallet?.holdings_count ?? 0)}
              angelConnected={angelConnected}
              onRefresh={() => { refresh(); apiFetch<VirtualWallet>('/api/v1/wallet').then(w => { if (w) setVirtualWallet(w); }); }}
            />

            {/* Run selector */}
            {allRuns.length > 0 && (
              <div className="relative group">
                <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-card hover:bg-muted/50 text-xs transition-colors">
                  <History className="h-3.5 w-3.5 text-muted-foreground" />
                  <span className="font-mono text-muted-foreground">
                    {selectedRunId
                      ? selectedRunId.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2}).*/, '$3/$2 $4:$5')
                      : 'Latest'}
                  </span>
                  <ChevronDown className="h-3 w-3 text-muted-foreground" />
                </button>
                <div className="absolute right-0 top-full mt-1 z-50 hidden group-hover:block w-64 rounded-xl border border-border bg-popover shadow-xl overflow-hidden">
                  <div className="p-1.5">
                    <button
                      onClick={() => selectRun(null)}
                      className={cn('w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs hover:bg-muted/50 transition-colors',
                        !selectedRunId && 'bg-primary/10 text-primary'
                      )}>
                      <span className="font-medium">Latest run</span>
                      {!selectedRunId && <CheckCircle2 className="h-3 w-3" />}
                    </button>
                    <div className="my-1 border-t border-border" />
                    <div className="max-h-64 overflow-y-auto space-y-0.5">
                      {allRuns.map(r => (
                        <button key={r.run_id} onClick={() => selectRun(r.run_id)}
                          className={cn('w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs hover:bg-muted/50 transition-colors',
                            selectedRunId === r.run_id && 'bg-primary/10 text-primary'
                          )}>
                          <div className="text-left">
                            <div className="font-mono font-medium">
                              {r.run_id.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/, '$3/$2/$1 $4:$5')}
                            </div>
                            <div className="text-[10px] text-muted-foreground">
                              {r.n_stocks ?? '?'} stocks{r.avg_accuracy ? ` · ${(r.avg_accuracy * 100).toFixed(1)}% acc` : ''}
                            </div>
                          </div>
                          {selectedRunId === r.run_id && <CheckCircle2 className="h-3 w-3 flex-shrink-0" />}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="p-2 rounded-lg border border-border hover:bg-muted/50 transition-colors">
              {mounted && (theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />)}
            </button>
          </div>
        </div>
      </header>

      {/* ── Live ticker ── */}
      <div className="border-b border-border bg-muted/5 flex-shrink-0">
        <div className="page-container py-2">
          <LivePriceTicker symbols={watchlist.length > 0 ? watchlist : undefined} prices={prices} change={change} connected={connected} />
        </div>
      </div>

      {/* ── Nav: single row, mode toggle on right ── */}
      <nav className="border-b border-border bg-background flex-shrink-0">
        <div className="page-container flex items-center -mb-px">
          {/* Sub-tabs — left side */}
          <div className="flex flex-1 overflow-x-auto scrollbar-none">
            {(viewMode === 'trade' ? TRADE_TABS : RESEARCH_TABS).map(tab => {
              const isActive = viewMode === 'trade' ? tradeTab === tab.id : researchTab === tab.id;
              return (
                <button key={tab.id}
                  onClick={() => viewMode === 'trade'
                    ? setTradeTab(tab.id as TradeTab)
                    : setResearchTab(tab.id as ResearchTab)}
                  className={cn(
                    'flex items-center gap-2 px-5 py-3.5 text-sm font-medium border-b-2 transition-all whitespace-nowrap',
                    isActive
                      ? 'border-primary text-primary'
                      : 'border-transparent text-muted-foreground hover:text-foreground hover:border-border'
                  )}>
                  {tab.label}
                  {tab.id === 'signals' && upCount > 0 && (
                    <span className="px-1.5 py-0.5 rounded-full bg-primary/20 text-primary text-[10px]">{upCount}</span>
                  )}
                  {tab.id === 'orders' && (dashboard?.orders?.count ?? 0) > 0 && (
                    <span className="px-1.5 py-0.5 rounded-full bg-yellow-500/20 text-yellow-400 text-[10px]">{dashboard?.orders?.count}</span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Mode toggle pill — right side */}
          <div className="flex items-center gap-0.5 ml-4 mb-0.5 bg-muted/40 rounded-lg p-0.5 flex-shrink-0">
            {([
              { id: 'trade',    label: 'Trade',    icon: Zap          },
              { id: 'research', label: 'Research', icon: FlaskConical },
            ] as { id: ViewMode; label: string; icon: any }[]).map(m => (
              <button key={m.id} onClick={() => setViewMode(m.id)}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all',
                  viewMode === m.id
                    ? 'bg-background text-foreground shadow-sm border border-border'
                    : 'text-muted-foreground hover:text-foreground'
                )}>
                <m.icon className="h-3 w-3" />
                {m.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* ── Main ── */}
      <main className="flex-1 page-container py-5">

        {/* ════════════════════════════════════════════════════
            TRADE VIEW
        ════════════════════════════════════════════════════ */}

        {/* TRADE / OVERVIEW */}
        {viewMode === 'trade' && tradeTab === 'overview' && (
          <div className="space-y-5">
            {selectedRunId && <RunContextBanner runId={selectedRunId} onClear={() => selectRun(null)} />}

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
              {/* Left: signals + pipeline */}
              <div className="xl:col-span-2 space-y-4">
                {/* 4 actionable stat cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <StatCard label="UP signals"    value={upCount}
                    sub={`${dashboard?.predictions?.down ?? 0} DOWN`} icon={TrendingUp} trend="up" />
                  <StatCard label="Down signals"  value={dashboard?.predictions?.down ?? 0}
                    sub="bearish" icon={TrendingDown} trend="down" />
                  <StatCard label="Deployed"
                    value={deployed > 0 ? `₹${(deployed / 100000).toFixed(1)}L` : '—'}
                    sub="pending orders" icon={Wallet} />
                  <StatCard label="Portfolio P&L"
                    value={
                      (angelConnected ? holdings !== null : virtualWallet !== null)
                        ? totalPnl === 0
                          ? '₹0'
                          : `${totalPnl > 0 ? '+' : ''}₹${Math.abs(totalPnl).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
                        : '—'
                    }
                    sub="unrealized" trend={totalPnl > 0 ? 'up' : totalPnl < 0 ? 'down' : 'neutral'} />
                </div>

                {/* Top UP signals list */}
                {predictions.filter(p => p.direction === 'UP').length > 0 && (
                  <div className="rounded-xl border border-border bg-card p-4">
                    <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-green-400" /> Top UP Signals Today
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {predictions
                        .filter(p => p.direction === 'UP')
                        .sort((a, b) => (b.prob_up ?? 0) - (a.prob_up ?? 0))
                        .slice(0, 10)
                        .map(p => (
                          <div key={p.symbol}
                            className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-muted/20 cursor-pointer transition-colors border border-border/50"
                            onClick={() => { setChartSym(p.symbol); setTradeTab('chart'); }}>
                            <span className="font-mono font-bold w-24 truncate">{p.symbol}</span>
                            <div className="flex-1 bg-muted/30 rounded-full h-1.5">
                              <div className="h-1.5 rounded-full bg-gradient-to-r from-green-500 to-emerald-400"
                                style={{ width: `${Math.min((p.prob_up ?? 0.5) * 150, 100)}%` }} />
                            </div>
                            <span className={cn('font-mono font-bold text-xs w-12 text-right',
                              (p.prob_up ?? 0) >= 0.58 ? 'text-green-400' : 'text-yellow-400'
                            )}>{((p.prob_up ?? 0) * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                    </div>
                    <button onClick={() => setTradeTab('signals')}
                      className="mt-3 text-xs text-muted-foreground hover:text-foreground transition-colors">
                      View all {upCount} signals →
                    </button>
                  </div>
                )}

                {/* Pipeline control */}
                <PipelineControl onComplete={onPipelineComplete} />
              </div>

              {/* Right: pipeline status + holdings */}
              <div className="space-y-4">
                {/* Pipeline status card */}
                <div className="rounded-xl border border-border bg-card p-4">
                  <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
                    <Activity className="h-4 w-4 text-primary" /> Pipeline Status
                  </h3>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-xs mb-1.5">
                        <span className="text-muted-foreground">Training progress</span>
                        <span className={cn('font-mono font-bold', isComplete ? 'text-green-400' : '')}>
                          {nStocks}/{nTotal} {isComplete ? '✓' : ''}
                        </span>
                      </div>
                      <div className="w-full bg-muted/30 rounded-full h-2">
                        <div className={cn('h-2 rounded-full transition-all duration-700',
                          isComplete ? 'bg-gradient-to-r from-green-500 to-emerald-400' : 'bg-gradient-to-r from-indigo-500 to-purple-500'
                        )} style={{ width: `${nTotal > 0 ? Math.min((nStocks / nTotal) * 100, 100) : 0}%` }} />
                      </div>
                    </div>
                    {[
                      ['Run ID',       dashboard?.run_id ?? '—'],
                      ['Avg accuracy', `${(avgAcc * 100).toFixed(2)}%`],
                      ['Sentiment',    `${dashboard?.sentiment?.n_symbols ?? 0} sym × ${dashboard?.sentiment?.n_dates ?? 0} days`],
                      ['Last update',  dashboard?.sentiment?.latest_date?.split(' ')[0] ?? '—'],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between text-xs pt-1">
                        <span className="text-muted-foreground">{k}</span>
                        <span className={cn('font-mono', k === 'Avg accuracy' && avgAcc >= 0.52 ? 'text-green-400 font-bold' : '')}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Cash + invested */}
                <div className="grid grid-cols-2 gap-3">
                  <StatCard label="Cash"
                    value={cash > 0 ? `₹${cash.toLocaleString('en-IN', { maximumFractionDigits: 0 })}` : '—'}
                    sub={angelConnected ? 'Angel One' : 'Virtual wallet'} icon={Wallet} trend="neutral" />
                  <StatCard label="Avg OOS Acc"
                    value={`${(avgAcc * 100).toFixed(1)}%`}
                    sub="walk-forward" icon={Activity}
                    trend={avgAcc >= 0.52 ? 'up' : avgAcc >= 0.50 ? 'neutral' : 'down'}
                    highlight={avgAcc >= 0.54} />
                </div>

                {/* Holdings */}
                {holdings && holdings.holdings.length > 0 && (
                  <div className="rounded-xl border border-border bg-card p-4">
                    <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <Wallet className="h-4 w-4 text-primary" /> Holdings
                    </h3>
                    <div className="space-y-1.5">
                      {holdings.holdings.slice(0, 6).map((h: any) => (
                        <div key={h.symbol} className="flex items-center justify-between text-xs">
                          <span className="font-mono font-bold w-24 truncate">{h.symbol}</span>
                          <span className="text-muted-foreground">{h.qty} sh</span>
                          <span className={cn('font-mono font-semibold', h.pnl >= 0 ? 'text-green-400' : 'text-red-400')}>
                            {h.pnl >= 0 ? '+' : ''}₹{Math.abs(h.pnl).toFixed(0)}
                          </span>
                        </div>
                      ))}
                      {holdings.holdings.length > 6 && (
                        <button onClick={() => setTradeTab('portfolio')}
                          className="text-xs text-muted-foreground hover:text-foreground mt-1">
                          +{holdings.holdings.length - 6} more →
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TRADE / SIGNALS */}
        {viewMode === 'trade' && tradeTab === 'signals' && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted-foreground">
              <div className="flex flex-wrap items-center gap-4">
                <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-green-400"/>{upCount} UP signals</span>
                <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-red-400"/>{dashboard?.predictions?.down ?? 0} DOWN signals</span>
                <span className="text-xs text-muted-foreground/60">Click symbol to open chart</span>
              </div>
              <button
                onClick={exportPredictionsCsv}
                disabled={!predictions.length}
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-xs font-medium text-foreground transition-colors hover:bg-muted/40 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Download className="h-3.5 w-3.5" />
                Export CSV
              </button>
            </div>
            <SignalsTable predictions={predictions} onSymbolClick={sym => { setChartSym(sym); setTradeTab('chart'); }} />
          </div>
        )}

        {/* TRADE / ORDERS */}
        {viewMode === 'trade' && tradeTab === 'orders' && (
          <OrdersPanel
            orders={dashboard?.orders?.orders ?? []}
            orderCount={dashboard?.orders?.count}
            totalDeployed={dashboard?.orders?.total_deployed}
            runId={dashboard?.run_id}
          />
        )}

        {/* TRADE / PORTFOLIO */}
        {viewMode === 'trade' && tradeTab === 'portfolio' && (
          <div className="space-y-5">
            <ExecutionPanel livePrices={prices} />
            <PortfolioPanel
              holdings={holdings} funds={funds} prices={prices}
              angelConnected={angelStatus?.credentials_present ?? false}
              onRefresh={refresh}
            />
          </div>
        )}

        {/* TRADE / CHART */}
        {viewMode === 'trade' && tradeTab === 'chart' && (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {enrichedStocks.map((s: any) => (
                <button key={s.symbol} onClick={() => setChartSym(s.symbol)}
                  className={cn(
                    'flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs transition-all',
                    chartSym === s.symbol
                      ? 'bg-indigo-500 text-white border-indigo-500 shadow shadow-indigo-500/30'
                      : 'border-border hover:bg-muted/50'
                  )}>
                  <span className="font-mono font-bold">{s.symbol}</span>
                  {s.oos_accuracy > 0 && (
                    <span className={cn('text-[9px]',
                      chartSym === s.symbol ? 'text-white/70' : s.oos_accuracy >= 0.54 ? 'text-green-400' : 'text-muted-foreground'
                    )}>{(s.oos_accuracy * 100).toFixed(0)}%</span>
                  )}
                  {s.direction && (
                    <span className={cn('text-[9px]',
                      s.direction === 'UP' ? chartSym === s.symbol ? 'text-green-200' : 'text-green-400' : chartSym === s.symbol ? 'text-red-200' : 'text-red-400'
                    )}>{s.direction === 'UP' ? '▲' : '▼'}</span>
                  )}
                </button>
              ))}
            </div>
            <CandlestickChart data={ohlcv} signals={chartSignals} symbol={chartSym} height={560} />
            {(() => {
              const s = enrichedStocks.find((x: any) => x.symbol === chartSym);
              const ltp = prices[chartSym]; const chg = change(chartSym);
              if (!s) return null;
              return (
                <div className="grid grid-cols-3 sm:grid-cols-5 lg:grid-cols-7 gap-3">
                  {[
                    { label: 'LTP',        value: ltp ? `₹${ltp.toFixed(2)}` : '—',              color: chg !== null ? (chg >= 0 ? 'text-green-400' : 'text-red-400') : '' },
                    { label: '1d Change',  value: chg !== null ? `${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%` : '—', color: chg !== null ? (chg >= 0 ? 'text-green-400' : 'text-red-400') : '' },
                    { label: 'OOS Acc',    value: `${((s.oos_accuracy || 0) * 100).toFixed(1)}%`, color: s.oos_accuracy >= 0.54 ? 'text-green-400' : '' },
                    { label: 'Best Model', value: s.best_model || '—',                             color: '' },
                    { label: 'Signal',     value: s.direction || 'None',                           color: s.direction === 'UP' ? 'text-green-400' : s.direction ? 'text-red-400' : '' },
                    { label: 'P(UP)',      value: s.prob_up ? `${(s.prob_up * 100).toFixed(1)}%` : '—', color: (s.prob_up || 0) >= 0.52 ? 'text-green-400' : '' },
                    { label: 'Pred Range', value: s.range_low != null && s.range_high != null ? `₹${s.range_low.toFixed(0)} - ₹${s.range_high.toFixed(0)}` : '—', color: '' },
                    { label: 'Move', value: s.predicted_move_pct != null ? `${s.predicted_move_pct > 0 ? '+' : ''}${s.predicted_move_pct.toFixed(2)}%` : '—', color: (s.predicted_move_pct || 0) > 0 ? 'text-green-400' : (s.predicted_move_pct || 0) < 0 ? 'text-red-400' : '' },
                    { label: 'Pred For', value: s.prediction_for || '—', color: '' },
                    { label: 'Predictions', value: s.n_predictions || '—',                         color: '' },
                  ].map(item => (
                    <div key={item.label} className="rounded-xl border border-border bg-card p-3">
                      <div className="text-[10px] text-muted-foreground uppercase tracking-wide">{item.label}</div>
                      <div className={cn('text-lg font-bold font-mono mt-0.5', item.color)}>{item.value}</div>
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>
        )}

        {/* ════════════════════════════════════════════════════
            RESEARCH VIEW
        ════════════════════════════════════════════════════ */}

        {/* RESEARCH / ACCURACY */}
        {viewMode === 'research' && researchTab === 'accuracy' && (
          <div className="space-y-3">
            {selectedRunId && <RunContextBanner runId={selectedRunId} onClear={() => selectRun(null)} />}
            {summary
              ? <AccuracyHeatmap stocks={summary.stocks} />
              : <div className="text-center py-16 text-muted-foreground text-sm">No run data yet — run the pipeline first.</div>}
          </div>
        )}

        {/* RESEARCH / ANALYSIS */}
        {viewMode === 'research' && researchTab === 'analysis' && (
          <div className="space-y-6">
            {selectedRunId && <RunContextBanner runId={selectedRunId} onClear={() => selectRun(null)} />}
            <OOSMetricsPanel summary={summary} capital={config.capital} />
            <BacktestPanel runId={dashboard?.run_id} onLoaded={setSharedBacktest} />
            <SentimentPanel />
            <PaperTradingPanel onLoaded={setSharedPaperSession} />
            <ResearchChecklist
              stocks={summaryStocks}
              backtest={sharedBacktest}
              paperSession={sharedPaperSession}
            />
          </div>
        )}

        {/* RESEARCH / RUNS */}
        {viewMode === 'research' && researchTab === 'runs' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold flex items-center gap-2">
                <History className="h-4 w-4 text-primary" /> Pipeline Run History
              </h2>
              <span className="text-xs text-muted-foreground">{allRuns.length} runs</span>
            </div>
            {allRuns.length === 0 ? (
              <div className="text-center py-16 text-muted-foreground text-sm">No runs yet — run the pipeline first.</div>
            ) : (
              <div className="rounded-xl border border-border bg-card overflow-hidden divide-y divide-border/50">
                {allRuns.map(r => (
                  <button key={r.run_id}
                    onClick={() => { selectRun(r.run_id); setResearchTab('accuracy'); }}
                    className={cn(
                      'w-full flex items-center justify-between px-5 py-4 hover:bg-muted/30 transition-colors text-left',
                      selectedRunId === r.run_id && 'bg-primary/5'
                    )}>
                    <div>
                      <div className="font-mono font-semibold text-sm">
                        {r.run_id.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/, '$3 $2 $1 · $4:$5')}
                      </div>
                      <div className="text-xs text-muted-foreground mt-0.5">
                        {r.n_stocks ?? '?'} stocks trained
                        {r.avg_accuracy ? ` · ${(r.avg_accuracy * 100).toFixed(1)}% avg accuracy` : ''}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {r.avg_accuracy && (
                        <span className={cn('text-sm font-mono font-bold',
                          r.avg_accuracy >= 0.54 ? 'text-green-400' : r.avg_accuracy >= 0.50 ? 'text-yellow-400' : 'text-red-400'
                        )}>
                          {(r.avg_accuracy * 100).toFixed(1)}%
                        </span>
                      )}
                      {selectedRunId === r.run_id
                        ? <CheckCircle2 className="h-4 w-4 text-primary" />
                        : <span className="text-xs text-muted-foreground">View →</span>}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

      </main>

      <footer className="border-t border-border flex-shrink-0">
        <div className="page-container py-2 flex items-center justify-between text-[10px] text-muted-foreground/40">
          <span>AlgoTrading V3 · NSE Walk-Forward Ensemble · Angel One SmartAPI</span>
          <span className="flex items-center gap-1.5">
            <span className={cn('w-1.5 h-1.5 rounded-full', connected ? 'bg-green-400 animate-pulse' : 'bg-muted-foreground')} />
            {connected ? 'Live prices connected' : 'Prices offline'}
          </span>
        </div>
      </footer>
    </div>
  );
}

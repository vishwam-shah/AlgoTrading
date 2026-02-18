'use client';
import * as React from 'react';

import { useState, useEffect, useCallback } from 'react';
import Select from 'react-select';
import { useTheme } from 'next-themes';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

// Import modular components
import { Header } from '@/components/Header';
import { Metrics } from '@/components/Metrics';
import { ProgressComponent } from '@/components/Progress';
import { Signals } from '@/components/Signals';
import { Chart } from '@/components/Chart';
import { StockSelector } from '@/components/StockSelector';
import { PerStockMetricsTable } from '@/components/PerStockMetricsTable';
import { SettingsModal } from '@/components/SettingsModal';
import { WalletOverview } from '@/components/WalletOverview';
import { FeatureImportance } from '@/components/FeatureImportance';
import { CSVExporter } from '@/components/CSVExporter';
import { Toaster, toast } from 'sonner';

// ==================== TYPES ====================

interface PipelineResult {
  backtest_results: Record<string, any>;
  signals: Record<string, any>;
  allocation: any;
  equity_curve: any[];
  feature_importance?: any[]; // Added for type safety
}


interface WalletData {
  balance: number;
  portfolio_value: number;
  total_value: number;
  total_pnl: number;
  total_pnl_pct: number;
  holdings_count: number;
  portfolio: Record<string, any>;
}

// ==================== MAIN DASHBOARD ====================

export default function Dashboard() {
    // Manual trade form state
    const [tradeSymbol, setTradeSymbol] = useState('');
    const [tradeAction, setTradeAction] = useState<'buy' | 'sell'>('buy');
    const [tradeQty, setTradeQty] = useState(1);
    const [tradeLoading, setTradeLoading] = useState(false);
    const [stockOptions, setStockOptions] = useState<{ value: string; label: string }[]>([]);
    const [selectedStock, setSelectedStock] = useState<{ value: string; label: string } | null>(null);
    const [stockPrice, setStockPrice] = useState<number | null>(null);
    // Cast react-select to a React component type to avoid TSX type mismatch
    const TypedSelect = Select as unknown as React.ComponentType<any>;
        // Fetch all stock tickers for dropdown
        useEffect(() => {
          const fetchStocks = async () => {
            try {
              const response = await fetch('http://localhost:8000/api/v1/stocks');
              const data = await response.json();
              if (data.stocks) {
                const options = data.stocks.map((s: any) => ({ value: s.symbol, label: `${s.symbol} - ${s.name || ''}` }));
                setStockOptions(options);
              }
            } catch (err) {
              setStockOptions([]);
            }
          };
          fetchStocks();
        }, []);

        // Fetch real-time price when stock changes
        useEffect(() => {
          if (!selectedStock) { setStockPrice(null); return; }
          const fetchPrice = async () => {
            try {
              const response = await fetch(`http://localhost:8000/api/v1/price/${selectedStock.value}`);
              const data = await response.json();
              setStockPrice(data.price ?? null);
            } catch {
              setStockPrice(null);
            }
          };
          fetchPrice();
        }, [selectedStock]);
    // Manual trade handler
    const handleManualTrade = async (e: React.FormEvent) => {
      e.preventDefault();
      if (!selectedStock || !selectedStock.value || !tradeQty || tradeQty <= 0) {
        toast.error('Please select a stock and enter a valid quantity.');
        return;
      }
      setTradeLoading(true);
      try {
        const response = await fetch('http://localhost:8000/api/v1/wallet/trade', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            symbol: selectedStock.value,
            action: tradeAction,
            quantity: tradeQty,
          }),
        });
        const data = await response.json();
        if (response.ok) {
          toast.success(`${tradeAction === 'buy' ? 'Bought' : 'Sold'} ${tradeQty} ${selectedStock.value}`);
          // Refresh wallet
          const walletResponse = await fetch('http://localhost:8000/api/v1/wallet');
          const walletData = await walletResponse.json();
          setWallet(walletData);
        } else {
          toast.error(data.detail || 'Trade failed');
        }
      } catch (err: any) {
        toast.error('Trade failed: ' + err.message);
      } finally {
        setTradeLoading(false);
      }
    };
  const { theme, setTheme } = useTheme();

  // State
  // (removed duplicate selectedStock)
  const [selectedSectors, setSelectedSectors] = useState<string[]>([]);
  const [selectedStocks, setSelectedStocks] = useState<string[]>([]);
  const [expandedSectors, setExpandedSectors] = useState<string[]>([]);
  const [sectorStockMap, setSectorStockMap] = useState<Record<string, string[]>>({});

  // Pipeline mode toggle
  const [pipelineMode, setPipelineMode] = useState<'engine' | 'v3'>('engine');

  // Pipeline state
  const [isRunning, setIsRunning] = useState(false);
  const [pipelineJobId, setPipelineJobId] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<any>(null);
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | null>(null);

  // Trading state
  const [isTrading, setIsTrading] = useState(false);
  const [wallet, setWallet] = useState<WalletData | null>(null);

  // UI state
  const [showSettingsModal, setShowSettingsModal] = useState(false);

  // Fetch stocks from backend
  useEffect(() => {
    const fetchStocks = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/stocks');
        const data = await response.json();

        // Transform list to sector map
        if (data.stocks) {
          const map: Record<string, string[]> = {};
          data.stocks.forEach((stock: any) => {
            if (!map[stock.sector]) {
              map[stock.sector] = [];
            }
            map[stock.sector].push(stock.symbol);
          });
          setSectorStockMap(map);
        }
      } catch (err) {
        console.error('Failed to fetch stocks:', err);
      }
    };
    fetchStocks();
  }, []);

  // Fetch wallet
  useEffect(() => {
    const fetchWallet = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/wallet');
        const data = await response.json();
        setWallet(data);
      } catch (err) {
        console.error('Failed to fetch wallet:', err);
      }
    };
    fetchWallet();
  }, []);

  // Poll pipeline status (handles both Engine and V3 modes)
  useEffect(() => {
    if (!isRunning || !pipelineJobId) return;

    let isMounted = true;

    const poll = async () => {
      try {
        const statusUrl = pipelineMode === 'v3'
          ? `http://localhost:8000/api/v1/v3/${pipelineJobId}/status`
          : `http://localhost:8000/api/v1/pipeline/${pipelineJobId}/status`;

        const response = await fetch(statusUrl);
        if (!isMounted) return;

        const data = await response.json();
        if (!isMounted) return;

        setPipelineStatus(data);

        if (data.status === 'completed') {
          setIsRunning(false);
          setPipelineResult(data.result);
          toast.success(`${pipelineMode === 'v3' ? 'V3' : 'Engine'} pipeline completed!`);
        } else if (data.status === 'failed') {
          setIsRunning(false);
          toast.error(`Pipeline failed: ${data.message || data.error || 'Unknown error'}`);
        } else {
          setTimeout(poll, 2000);
        }
      } catch (err) {
        console.error('Failed to poll pipeline status:', err);
        if (isMounted) setTimeout(poll, 2000);
      }
    };

    poll();

    return () => { isMounted = false; };
  }, [isRunning, pipelineJobId, pipelineMode]);

  // Handlers
  const handleRunPipeline = async () => {
    if (isRunning) return;

    try {
      setIsRunning(true);
      setPipelineResult(null);
      setPipelineStatus(null);

      const stocksToRun = [...selectedStocks];

      if (stocksToRun.length === 0) {
        toast.error('Please select at least one stock to run.');
        setIsRunning(false);
        return;
      }

      const endpoint = pipelineMode === 'v3'
        ? 'http://localhost:8000/api/v1/v3/run'
        : 'http://localhost:8000/api/v1/pipeline/run';

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: stocksToRun }),
      });

      const data = await response.json();
      setPipelineJobId(data.job_id);
      toast.success(`${pipelineMode === 'v3' ? 'V3' : 'Engine'} pipeline started`);
    } catch (err: any) {
      setIsRunning(false);
      toast.error(`Failed to start pipeline: ${err.message}`);
    }
  };

  const handleExecuteTrade = async (symbol: string) => {
    try {
      setIsTrading(true);
      const signal = pipelineResult?.signals[symbol];
      if (!signal) return;

      const response = await fetch('http://localhost:8000/api/v1/wallet/trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol,
          action: signal.action,
          quantity: 1,
        }),
      });

      if (response.ok) {
        toast.success(`Trade for ${symbol} executed successfully! 💰`);
        // Refresh wallet
        const walletResponse = await fetch('http://localhost:8000/api/v1/wallet');
        const walletData = await walletResponse.json();
        setWallet(walletData);
      } else {
        throw new Error('Trade execution failed');
      }
    } catch (err: any) {
      toast.error(`Trade failed: ${err.message}`);
    } finally {
      setIsTrading(false);
    }
  };

  const toggleSector = (sector: string) => {
    // Handle "Select All" case
    if (sector === '__ALL__') {
      const allSectors = Object.keys(sectorStockMap);
      if (selectedSectors.length === allSectors.length) {
        // Deselect everything
        setSelectedSectors([]);
        setSelectedStocks([]);
        setExpandedSectors([]);
      } else {
        // Select everything
        setSelectedSectors(allSectors);
        const allStocks = Object.values(sectorStockMap).flat();
        setSelectedStocks(allStocks);
        // Do not auto-expand all to avoid UI clutter
      }
      return;
    }

    const isSelected = selectedSectors.includes(sector);
    const sectorStocks = sectorStockMap[sector] || [];

    if (isSelected) {
      // Deselect sector and all its stocks
      setSelectedSectors(prev => prev.filter(s => s !== sector));
      setSelectedStocks(prev => prev.filter(s => !sectorStocks.includes(s)));
    } else {
      // Select sector and all its stocks
      setSelectedSectors(prev => [...prev, sector]);
      setSelectedStocks(prev => {
        const newStocks = [...prev];
        sectorStocks.forEach(s => {
          if (!newStocks.includes(s)) newStocks.push(s);
        });
        return newStocks;
      });
      // Auto-expand the sector to show selected stocks
      setExpandedSectors(prev => prev.includes(sector) ? prev : [...prev, sector]);
    }
  };

  const toggleStock = (stock: string) => {
    setSelectedStocks(prev =>
      prev.includes(stock)
        ? prev.filter(s => s !== stock)
        : [...prev, stock]
    );
  };

  const toggleSectorExpand = (sector: string) => {
    setExpandedSectors(prev =>
      prev.includes(sector)
        ? prev.filter(s => s !== sector)
        : [...prev, sector]
    );
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header Component */}
      <Header
        theme={theme}
        onThemeToggle={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        onSettingsClick={() => setShowSettingsModal(true)}
        pipelineResult={pipelineResult}
        pipelineJobId={pipelineJobId}
        wallet={wallet}
      />

      {/* Progress Component */}
      <ProgressComponent
        isRunning={isRunning}
        pipelineStatus={pipelineStatus}
      />

      <Toaster position="top-right" theme={theme === 'dark' ? 'dark' : 'light'} />

      <main className="container mx-auto px-4 py-6 space-y-6">
        {/* Manual Trade Form */}
        <form onSubmit={handleManualTrade} className="mb-6 flex flex-wrap gap-4 items-end bg-card border border-border rounded-lg p-4">
          <div className="min-w-[220px]">
            <label className="block text-xs mb-1 font-medium">Stock</label>
            <TypedSelect
              options={stockOptions}
              value={selectedStock}
              onChange={setSelectedStock}
              isClearable
              isSearchable
              placeholder="Select stock..."
              classNamePrefix="react-select"
              instanceId="manual-stock-select"
              inputId="manual-stock-select-input"
              styles={{ menu: (base: any) => ({ ...base, zIndex: 9999 }) }}
            />
          </div>
          <div>
            <label className="block text-xs mb-1 font-medium">Action</label>
            <select value={tradeAction} onChange={e => setTradeAction(e.target.value as 'buy' | 'sell')} className="input input-bordered w-20">
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
          </div>
          <div>
            <label className="block text-xs mb-1 font-medium">Quantity</label>
            <input type="number" min={1} value={tradeQty} onChange={e => setTradeQty(Number(e.target.value))} className="input input-bordered w-20" />
          </div>
          <div>
            <label className="block text-xs mb-1 font-medium">Price</label>
            <div className="font-mono text-base min-w-[80px]">{stockPrice !== null ? `₹${stockPrice.toFixed(2)}` : '--'}</div>
          </div>
          <button type="submit" className="btn" style={{ background: 'black', color: 'white' }} disabled={tradeLoading}>
            {tradeLoading ? 'Processing...' : 'Submit'}
          </button>
        </form>
        {/* Metrics Cards */}
        <Metrics pipelineResult={pipelineResult} wallet={wallet} />

        {/* Pipeline Mode Toggle */}
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-muted-foreground">Pipeline:</span>
          <div className="flex gap-1 bg-secondary rounded-lg p-1">
            <button
              onClick={() => setPipelineMode('engine')}
              className={cn(
                'px-3 py-1.5 rounded-md text-sm font-medium transition-all',
                pipelineMode === 'engine'
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              Engine
            </button>
            <button
              onClick={() => setPipelineMode('v3')}
              className={cn(
                'px-3 py-1.5 rounded-md text-sm font-medium transition-all',
                pipelineMode === 'v3'
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              V3 Model
            </button>
          </div>
          {pipelineMode === 'v3' && (
            <span className="text-xs text-muted-foreground">Regression pipeline with 5 window configs</span>
          )}
        </div>

        {/* Stock Selector */}
        <StockSelector
          sectorStockMap={sectorStockMap}
          selectedSectors={selectedSectors}
          selectedStocks={selectedStocks}
          expandedSectors={expandedSectors}
          onSectorToggle={toggleSector}
          onStockToggle={toggleStock}
          onExpandSector={toggleSectorExpand}
          onRunPipeline={handleRunPipeline}
          isRunning={isRunning}
        />

        {/* Results Section */}
        {pipelineResult && (
          <div className="space-y-6">
            {/* Trading Signals */}
            <div className="bg-card rounded-lg border border-border p-6">
              <h2 className="text-lg font-semibold mb-4">Trading Signals</h2>
              <Signals
                signals={pipelineResult.signals || {}}
                onExecuteTrade={handleExecuteTrade}
                isTrading={isTrading}
              />
            </div>

            {/* Equity Curve & Feature Importance */}
            {pipelineResult.equity_curve && pipelineResult.equity_curve.length > 0 && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 bg-card rounded-lg border border-border p-6">
                  <h2 className="text-lg font-semibold mb-4">Portfolio Performance</h2>
                  <Chart equityCurve={pipelineResult.equity_curve} />
                </div>
                <div className="bg-card rounded-lg border border-border p-6">
                  <FeatureImportance
                    featureImportance={Object.values(pipelineResult.backtest_results)[0]?.feature_importance || []}
                  />
                </div>
              </div>
            )}

            {/* Wallet Overview */}
            <WalletOverview wallet={wallet} />

            {/* Per-Stock Metrics Table */}
            <div className="bg-card rounded-lg border border-border p-6">
              <h2 className="text-lg font-semibold mb-4">Per-Stock Performance Metrics</h2>
              <PerStockMetricsTable
                backtestResults={pipelineResult.backtest_results}
                signals={pipelineResult.signals}
              />
            </div>
          </div>
        )}

        {/* Empty State */}
        {!pipelineResult && !isRunning && (
          <div className="bg-secondary/20 rounded-lg border border-dashed border-border p-12 text-center">
            <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">No Results Yet</h3>
            <p className="text-sm text-muted-foreground">
              Select stocks or sectors above and click "Run Pipeline" to generate trading signals
            </p>
          </div>
        )}
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettingsModal}
        onClose={() => setShowSettingsModal(false)}
        onSave={(settings) => {
          console.log('Pipeline settings updated:', settings);
          // TODO: Wire this to backend API to update pipeline config
          setShowSettingsModal(false);
        }}
      />
    </div>
  );
}

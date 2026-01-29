'use client';

import { useState, useEffect, useCallback } from 'react';
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
  const { theme, setTheme } = useTheme();

  // State
  const [selectedStock, setSelectedStock] = useState('SBIN');
  const [selectedSectors, setSelectedSectors] = useState<string[]>([]);
  const [selectedStocks, setSelectedStocks] = useState<string[]>([]);
  const [expandedSectors, setExpandedSectors] = useState<string[]>([]);
  const [sectorStockMap, setSectorStockMap] = useState<Record<string, string[]>>({});

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

  // Poll pipeline status
  useEffect(() => {
    if (!isRunning || !pipelineJobId) return;

    let isMounted = true;

    const poll = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/v1/pipeline/${pipelineJobId}/status`);
        if (!isMounted) return;

        const data = await response.json();
        if (!isMounted) return;

        setPipelineStatus(data);

        if (data.status === 'completed') {
          setIsRunning(false);
          setPipelineResult(data.result);
          toast.success('Pipeline completed successfully! 🎉');
        } else if (data.status === 'failed') {
          setIsRunning(false);
          toast.error(`Pipeline failed: ${data.error || 'Unknown error'}`);
        } else {
          // Continue polling if still running
          setTimeout(poll, 2000);
        }
      } catch (err) {
        console.error('Failed to poll pipeline status:', err);
        // Retry on error if still mounted
        if (isMounted) setTimeout(poll, 2000);
      }
    };

    poll();

    return () => { isMounted = false; };
  }, [isRunning, pipelineJobId]);

  // Handlers
  const handleRunPipeline = async () => {
    if (isRunning) return;

    try {
      setIsRunning(true);
      setPipelineResult(null);
      setPipelineStatus(null);

      // Use selected stocks directly (selection logic is handled by UI handlers)
      const stocksToRun = [...selectedStocks];

      if (stocksToRun.length === 0) {
        toast.error('❌ Please select at least one stock to run.');
        setIsRunning(false);
        return;
      }

      const response = await fetch('http://localhost:8000/api/v1/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: stocksToRun }),
      });

      const data = await response.json();
      setPipelineJobId(data.job_id);
      toast.success('Pipeline started successfully 🚀');
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
        {/* Metrics Cards */}
        <Metrics pipelineResult={pipelineResult} wallet={wallet} />

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

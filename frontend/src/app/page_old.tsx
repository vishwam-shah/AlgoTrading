'use client';

import { useState, useEffect, useCallback } from 'react';
import { useTheme } from 'next-themes';
import * as Select from '@radix-ui/react-select';
import * as Progress from '@radix-ui/react-progress';
import * as Slider from '@radix-ui/react-slider';
import * as Switch from '@radix-ui/react-switch';
import * as Dialog from '@radix-ui/react-dialog';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Play,
  Activity,
  DollarSign,
  Target,
  BarChart3,
  Zap,
  Brain,
  AlertCircle,
  Sun,
  Moon,
  ChevronDown,
  Check,
  Loader2,
  Settings,
  LineChart,
  Briefcase,
  Layers,
  Cpu,
  Shield,
  ArrowUpRight,
  ArrowDownRight,
  X,
  Wallet,
  RefreshCw,
  ShoppingCart,
  MinusCircle,
  PlusCircle,
  History,
} from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import { cn } from '@/lib/utils';

// Types
interface BacktestResult {
  symbol: string;
  total_trades?: number;
  win_rate?: number;
  total_return?: number;
  max_drawdown?: number;
  sharpe_ratio?: number;
  profit_factor?: number;
  equity_curve?: { date: string; equity: number }[];
  trades?: TradeData[];
  error?: string;
}

interface TradeData {
  entry_date: string;
  exit_date: string;
  direction: string;
  entry_price: number;
  exit_price: number;
  shares: number;
  pnl: number;
  return_pct: number;
  exit_reason: string;
}

interface BacktestResponse {
  results: BacktestResult[];
  summary: {
    total_stocks?: number;
    profitable_stocks?: number;
    avg_return?: number;
    best_return?: number;
    worst_return?: number;
    avg_sharpe?: number;
    avg_win_rate?: number;
  };
  timestamp: string;
}

interface SentimentData {
  symbol: string;
  sentiment: {
    overall_sentiment: number;
    sentiment_label: string;
    news_volume: number;
    positive_ratio: number;
    negative_ratio: number;
    neutral_ratio: number;
  };
  timestamp: string;
}

interface WalletData {
  balance: number;
  initial_balance: number;
  portfolio_value: number;
  total_value: number;
  total_pnl: number;
  total_pnl_pct: number;
  realized_pnl: number;
  unrealized_pnl: number;
  portfolio: Record<string, PortfolioHolding>;
  holdings_count: number;
}

interface PortfolioHolding {
  shares: number;
  avg_price: number;
  current_price: number;
  current_value: number;
  pnl: number;
  pnl_pct: number;
  bought_at: string;
}

interface Transaction {
  type: string;
  symbol: string;
  quantity: number;
  price: number;
  total: number;
  pnl?: number;
  timestamp: string;
}

interface Config {
  model_type: string;
  lookback_days: number;
  train_test_split: number;
  min_confidence: number;
  max_position_pct: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  use_sentiment: boolean;
  use_technical: boolean;
  use_volume: boolean;
}

interface MarketPrice {
  symbol: string;
  price: number;
  change: number;
  change_pct: number;
  high: number;
  low: number;
  volume: number;
}

const AVAILABLE_STOCKS = [
  { value: 'SBIN', label: 'State Bank of India', sector: 'Banking' },
  { value: 'AXISBANK', label: 'Axis Bank', sector: 'Banking' },
  { value: 'HDFCBANK', label: 'HDFC Bank', sector: 'Banking' },
  { value: 'ICICIBANK', label: 'ICICI Bank', sector: 'Banking' },
  { value: 'KOTAKBANK', label: 'Kotak Bank', sector: 'Banking' },
  { value: 'RELIANCE', label: 'Reliance Industries', sector: 'Energy' },
  { value: 'TCS', label: 'Tata Consultancy', sector: 'IT' },
  { value: 'INFY', label: 'Infosys', sector: 'IT' },
  { value: 'WIPRO', label: 'Wipro', sector: 'IT' },
  { value: 'HCLTECH', label: 'HCL Technologies', sector: 'IT' },
  { value: 'BHARTIARTL', label: 'Bharti Airtel', sector: 'Telecom' },
  { value: 'ITC', label: 'ITC Limited', sector: 'FMCG' },
  { value: 'HINDUNILVR', label: 'Hindustan Unilever', sector: 'FMCG' },
  { value: 'TATAMOTORS', label: 'Tata Motors', sector: 'Auto' },
  { value: 'MARUTI', label: 'Maruti Suzuki', sector: 'Auto' },
  { value: 'SUNPHARMA', label: 'Sun Pharma', sector: 'Pharma' },
];

const MODEL_TYPES = [
  { value: 'xgboost', label: 'XGBoost', description: 'Fast gradient boosting' },
  { value: 'lstm', label: 'LSTM', description: 'Deep learning sequence model' },
  { value: 'transformer', label: 'Transformer', description: 'Attention-based model' },
  { value: 'ensemble', label: 'Ensemble', description: 'Combined model approach' },
];

// Helpers
const safeToFixed = (val: number | undefined | null, decimals = 2): string => {
  if (val === undefined || val === null || isNaN(val)) return '0.00';
  return Number(val).toFixed(decimals);
};

const formatPercent = (val: number | undefined | null): string => {
  if (val === undefined || val === null || isNaN(val)) return '0.00%';
  return `${(Number(val) * 100).toFixed(2)}%`;
};

const formatCurrency = (val: number | undefined | null): string => {
  if (val === undefined || val === null || isNaN(val)) return '₹0';
  return `₹${Number(val).toLocaleString('en-IN')}`;
};

// Stat Card Component
const StatCard = ({ 
  icon: Icon, 
  label, 
  value, 
  subValue, 
  trend,
  color = 'primary' 
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  subValue?: string;
  trend?: 'up' | 'down' | 'neutral';
  color?: 'primary' | 'green' | 'red' | 'yellow';
}) => {
  const colors = {
    primary: 'text-primary bg-primary/10',
    green: 'text-green-500 bg-green-500/10',
    red: 'text-red-500 bg-red-500/10',
    yellow: 'text-yellow-500 bg-yellow-500/10',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-border/50 bg-card p-5 shadow-sm hover:shadow-md transition-all"
    >
      <div className="flex items-center gap-3 mb-3">
        <div className={cn('rounded-lg p-2', colors[color])}>
          <Icon className="h-5 w-5" />
        </div>
        <span className="text-sm text-muted-foreground">{label}</span>
      </div>
      <div className="flex items-end justify-between">
        <div>
          <span className={cn('text-2xl font-bold', {
            'text-green-500': trend === 'up',
            'text-red-500': trend === 'down',
          })}>{value}</span>
          {subValue && (
            <div className="flex items-center gap-1 mt-1">
              {trend === 'up' && <ArrowUpRight className="h-3 w-3 text-green-500" />}
              {trend === 'down' && <ArrowDownRight className="h-3 w-3 text-red-500" />}
              <span className="text-xs text-muted-foreground">{subValue}</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

// Configuration Panel Component
const ConfigPanel = ({ 
  config, 
  setConfig, 
  onApply 
}: { 
  config: Config; 
  setConfig: (c: Config) => void;
  onApply: () => void;
}) => {
  return (
    <div className="space-y-6">
      {/* Model Selection */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Model Type</label>
        <div className="grid grid-cols-2 gap-2">
          {MODEL_TYPES.map((model) => (
            <button
              key={model.value}
              onClick={() => setConfig({ ...config, model_type: model.value })}
              className={cn(
                'p-3 rounded-lg border text-left transition-all',
                config.model_type === model.value
                  ? 'border-primary bg-primary/5'
                  : 'border-border hover:border-primary/50'
              )}
            >
              <div className="font-medium text-sm">{model.label}</div>
              <div className="text-xs text-muted-foreground">{model.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Sliders */}
      <div className="space-y-4">
        <div className="space-y-2">
          <div className="flex justify-between">
            <label className="text-sm font-medium">Min Confidence</label>
            <span className="text-sm text-muted-foreground">{(config.min_confidence * 100).toFixed(0)}%</span>
          </div>
          <Slider.Root
            className="relative flex items-center select-none touch-none w-full h-5"
            value={[config.min_confidence * 100]}
            onValueChange={([v]) => setConfig({ ...config, min_confidence: v / 100 })}
            max={90}
            min={50}
            step={5}
          >
            <Slider.Track className="bg-secondary relative grow rounded-full h-2">
              <Slider.Range className="absolute bg-primary rounded-full h-full" />
            </Slider.Track>
            <Slider.Thumb className="block w-4 h-4 bg-primary rounded-full hover:bg-primary/90 focus:outline-none" />
          </Slider.Root>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <label className="text-sm font-medium">Max Position Size</label>
            <span className="text-sm text-muted-foreground">{(config.max_position_pct * 100).toFixed(0)}%</span>
          </div>
          <Slider.Root
            className="relative flex items-center select-none touch-none w-full h-5"
            value={[config.max_position_pct * 100]}
            onValueChange={([v]) => setConfig({ ...config, max_position_pct: v / 100 })}
            max={30}
            min={5}
            step={5}
          >
            <Slider.Track className="bg-secondary relative grow rounded-full h-2">
              <Slider.Range className="absolute bg-primary rounded-full h-full" />
            </Slider.Track>
            <Slider.Thumb className="block w-4 h-4 bg-primary rounded-full hover:bg-primary/90 focus:outline-none" />
          </Slider.Root>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <label className="text-sm font-medium">Stop Loss</label>
            <span className="text-sm text-muted-foreground">{(config.stop_loss_pct * 100).toFixed(0)}%</span>
          </div>
          <Slider.Root
            className="relative flex items-center select-none touch-none w-full h-5"
            value={[config.stop_loss_pct * 100]}
            onValueChange={([v]) => setConfig({ ...config, stop_loss_pct: v / 100 })}
            max={10}
            min={1}
            step={1}
          >
            <Slider.Track className="bg-secondary relative grow rounded-full h-2">
              <Slider.Range className="absolute bg-red-500 rounded-full h-full" />
            </Slider.Track>
            <Slider.Thumb className="block w-4 h-4 bg-red-500 rounded-full hover:bg-red-400 focus:outline-none" />
          </Slider.Root>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <label className="text-sm font-medium">Take Profit</label>
            <span className="text-sm text-muted-foreground">{(config.take_profit_pct * 100).toFixed(0)}%</span>
          </div>
          <Slider.Root
            className="relative flex items-center select-none touch-none w-full h-5"
            value={[config.take_profit_pct * 100]}
            onValueChange={([v]) => setConfig({ ...config, take_profit_pct: v / 100 })}
            max={20}
            min={2}
            step={1}
          >
            <Slider.Track className="bg-secondary relative grow rounded-full h-2">
              <Slider.Range className="absolute bg-green-500 rounded-full h-full" />
            </Slider.Track>
            <Slider.Thumb className="block w-4 h-4 bg-green-500 rounded-full hover:bg-green-400 focus:outline-none" />
          </Slider.Root>
        </div>
      </div>

      {/* Feature Toggles */}
      <div className="space-y-3">
        <label className="text-sm font-medium">Features</label>
        <div className="space-y-2">
          {[
            { key: 'use_sentiment', label: 'Sentiment Analysis', icon: Brain },
            { key: 'use_technical', label: 'Technical Indicators', icon: LineChart },
            { key: 'use_volume', label: 'Volume Analysis', icon: BarChart3 },
          ].map(({ key, label, icon: FeatureIcon }) => (
            <div key={key} className="flex items-center justify-between p-2 rounded-lg bg-secondary/30">
              <div className="flex items-center gap-2">
                <FeatureIcon className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm">{label}</span>
              </div>
              <Switch.Root
                checked={config[key as keyof Config] as boolean}
                onCheckedChange={(checked) => setConfig({ ...config, [key]: checked })}
                className={cn(
                  'w-10 h-5 rounded-full relative transition-colors',
                  config[key as keyof Config] ? 'bg-primary' : 'bg-muted'
                )}
              >
                <Switch.Thumb className={cn(
                  'block w-4 h-4 bg-white rounded-full transition-transform',
                  config[key as keyof Config] ? 'translate-x-5' : 'translate-x-0.5'
                )} />
              </Switch.Root>
            </div>
          ))}
        </div>
      </div>

      <button
        onClick={onApply}
        className="w-full py-2 px-4 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
      >
        Apply Configuration
      </button>
    </div>
  );
};

// Trade History Component
const TradeHistory = ({ trades }: { trades: TradeData[] }) => {
  if (!trades || trades.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No trades yet
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-2 px-2 font-medium text-muted-foreground">Entry</th>
            <th className="text-left py-2 px-2 font-medium text-muted-foreground">Exit</th>
            <th className="text-right py-2 px-2 font-medium text-muted-foreground">Entry ₹</th>
            <th className="text-right py-2 px-2 font-medium text-muted-foreground">Exit ₹</th>
            <th className="text-right py-2 px-2 font-medium text-muted-foreground">P&L</th>
            <th className="text-left py-2 px-2 font-medium text-muted-foreground">Reason</th>
          </tr>
        </thead>
        <tbody>
          {trades.slice(0, 10).map((trade, idx) => (
            <tr key={idx} className="border-b border-border/50 hover:bg-secondary/30">
              <td className="py-2 px-2">{trade.entry_date}</td>
              <td className="py-2 px-2">{trade.exit_date}</td>
              <td className="py-2 px-2 text-right">{formatCurrency(trade.entry_price)}</td>
              <td className="py-2 px-2 text-right">{formatCurrency(trade.exit_price)}</td>
              <td className={cn('py-2 px-2 text-right font-medium', trade.pnl >= 0 ? 'text-green-500' : 'text-red-500')}>
                {formatCurrency(trade.pnl)}
              </td>
              <td className="py-2 px-2">
                <span className={cn(
                  'px-2 py-0.5 rounded-full text-xs',
                  trade.exit_reason === 'TARGET' ? 'bg-green-500/10 text-green-500' :
                  trade.exit_reason === 'STOP_LOSS' ? 'bg-red-500/10 text-red-500' :
                  'bg-yellow-500/10 text-yellow-500'
                )}>
                  {trade.exit_reason}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default function Dashboard() {
  const { theme, setTheme } = useTheme();
  const [selectedStock, setSelectedStock] = useState('SBIN');
  const [selectedStocks] = useState<string[]>(['SBIN']);
  const [isRunning, setIsRunning] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [allResults, setAllResults] = useState<BacktestResponse | null>(null);
  const [sentiment, setSentiment] = useState<SentimentData | null>(null);
  const [marketPrice, setMarketPrice] = useState<MarketPrice | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [configOpen, setConfigOpen] = useState(false);
  const [walletOpen, setWalletOpen] = useState(false);
  const [wallet, setWallet] = useState<WalletData | null>(null);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [tradeQuantity, setTradeQuantity] = useState<number>(1);
  const [isTrading, setIsTrading] = useState(false);
  const [config, setConfig] = useState<Config>({
    model_type: 'xgboost',
    lookback_days: 500,
    train_test_split: 0.8,
    min_confidence: 0.55,
    max_position_pct: 0.15,
    stop_loss_pct: 0.03,
    take_profit_pct: 0.05,
    use_sentiment: true,
    use_technical: true,
    use_volume: true,
  });

  const currentStockInfo = AVAILABLE_STOCKS.find(s => s.value === selectedStock);

  // Fetch wallet
  const fetchWallet = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/wallet');
      if (res.ok) {
        const data = await res.json();
        setWallet(data);
      }
    } catch (err) {
      console.error('Failed to fetch wallet:', err);
    }
  }, []);

  // Fetch transactions
  const fetchTransactions = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/wallet/transactions?limit=20');
      if (res.ok) {
        const data = await res.json();
        setTransactions(data.transactions || []);
      }
    } catch (err) {
      console.error('Failed to fetch transactions:', err);
    }
  }, []);

  // Execute trade
  const executeTrade = async (action: 'buy' | 'sell') => {
    if (!marketPrice) return;
    setIsTrading(true);
    try {
      const res = await fetch('/api/v1/wallet/trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: selectedStock,
          action: action,
          quantity: tradeQuantity,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        await fetchWallet();
        await fetchTransactions();
        setError(null);
      } else {
        setError(data.detail || 'Trade failed');
      }
    } catch (err) {
      setError('Failed to execute trade');
    } finally {
      setIsTrading(false);
    }
  };

  // Auto-trade with AI signal
  const autoTrade = async () => {
    setIsTrading(true);
    setError(null);
    try {
      const res = await fetch(`/api/v1/wallet/auto-trade?symbol=${selectedStock}`, {
        method: 'POST',
      });
      const data = await res.json();
      if (res.ok) {
        await fetchWallet();
        await fetchTransactions();
        if (data.executed) {
          // Show success message with action taken
          const action = data.action_taken?.action || data.signal?.action;
          const qty = data.action_taken?.quantity || 0;
          setError(`✅ AI ${action}: ${qty} shares of ${selectedStock}`);
          setTimeout(() => setError(null), 3000);
        } else {
          // Show detailed reason with prediction value
          const pred = data.signal?.prediction;
          const action = data.signal?.action;
          const reason = data.reason || 'No action taken';
          if (pred !== undefined) {
            setError(`AI Signal: ${action} (prediction: ${(pred * 100).toFixed(1)}%) - ${reason}`);
          } else {
            setError(reason);
          }
        }
      } else {
        setError(data.detail || 'Auto-trade failed');
      }
    } catch (err) {
      setError('Failed to auto-trade');
    } finally {
      setIsTrading(false);
    }
  };

  // Reset wallet
  const resetWallet = async () => {
    try {
      await fetch('/api/v1/wallet/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ initial_balance: 100000 }),
      });
      await fetchWallet();
      await fetchTransactions();
    } catch (err) {
      setError('Failed to reset wallet');
    }
  };

  // Fetch sentiment
  const fetchSentiment = useCallback(async () => {
    try {
      const res = await fetch(`/api/v1/sentiment/${selectedStock}`);
      if (res.ok) {
        const data = await res.json();
        setSentiment(data);
      }
    } catch (err) {
      console.error('Failed to fetch sentiment:', err);
    }
  }, [selectedStock]);

  // Fetch market price
  const fetchMarketPrice = useCallback(async () => {
    try {
      const res = await fetch(`/api/v1/market/price/${selectedStock}`);
      if (res.ok) {
        const data = await res.json();
        setMarketPrice(data);
      }
    } catch (err) {
      console.error('Failed to fetch market price:', err);
    }
  }, [selectedStock]);

  // Auto-fetch on stock change
  useEffect(() => {
    fetchSentiment();
    fetchMarketPrice();
    fetchWallet();
    fetchTransactions();
  }, [selectedStock, fetchSentiment, fetchMarketPrice, fetchWallet, fetchTransactions]);

  // Run pipeline
  const runPipeline = async () => {
    setIsRunning(true);
    setProgress(0);
    setResult(null);
    setError(null);

    try {
      setProgress(10);
      await fetchSentiment();
      await fetchMarketPrice();

      setProgress(20);
      const res = await fetch('/api/v1/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbols: selectedStocks.length > 0 ? selectedStocks : [selectedStock],
          capital: 100000,
          days: config.lookback_days,
          config: config,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setJobId(data.job_id);
        pollBacktest(data.job_id);
      } else {
        const errData = await res.json();
        setError(errData.detail || 'Failed to start backtest');
        setIsRunning(false);
      }
    } catch (err) {
      console.error('Failed to start pipeline:', err);
      setError('Failed to connect to server');
      setIsRunning(false);
    }
  };

  // Poll backtest status
  const pollBacktest = async (id: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/backtest/${id}`);
        const data = await res.json();

        setProgress(20 + (data.progress || 0) * 0.8);

        if (data.status === 'completed') {
          const resultRes = await fetch(`/api/v1/results/${id}`);
          const resultData: BacktestResponse = await resultRes.json();
          setAllResults(resultData);

          const stockResult = resultData.results.find((r) => r.symbol === selectedStock);
          if (stockResult) {
            setResult(stockResult);
          } else if (resultData.results.length > 0) {
            setResult(resultData.results[0]);
          }

          setIsRunning(false);
          clearInterval(pollInterval);
        } else if (data.status === 'failed') {
          setError(data.message || 'Backtest failed');
          setIsRunning(false);
          clearInterval(pollInterval);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 2000);
  };

  // Apply config
  const applyConfig = async () => {
    try {
      await fetch('/api/v1/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      setConfigOpen(false);
    } catch (err) {
      console.error('Failed to apply config:', err);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Wallet Banner */}
      {wallet && (
        <div className="bg-gradient-to-r from-primary/10 via-primary/5 to-transparent border-b border-border/50">
          <div className="container mx-auto px-4 py-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                  <Wallet className="h-5 w-5 text-primary" />
                  <span className="text-sm text-muted-foreground">Balance:</span>
                  <span className="font-bold text-lg">{formatCurrency(wallet.balance)}</span>
                </div>
                <div className="h-6 w-px bg-border" />
                <div className="flex items-center gap-2">
                  <Briefcase className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Portfolio:</span>
                  <span className="font-semibold">{formatCurrency(wallet.portfolio_value)}</span>
                </div>
                <div className="h-6 w-px bg-border" />
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">Total:</span>
                  <span className="font-bold">{formatCurrency(wallet.total_value)}</span>
                  <span className={cn(
                    'text-sm font-medium px-2 py-0.5 rounded',
                    wallet.total_pnl >= 0 ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'
                  )}>
                    {wallet.total_pnl >= 0 ? '+' : ''}{formatCurrency(wallet.total_pnl)} ({wallet.total_pnl_pct.toFixed(2)}%)
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setWalletOpen(true)}
                  className="text-sm text-primary hover:underline flex items-center gap-1"
                >
                  <History className="h-4 w-4" />
                  View Details
                </button>
                <button
                  onClick={resetWallet}
                  className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
                >
                  <RefreshCw className="h-3 w-3" />
                  Reset
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border/50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center">
                <Brain className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold">AI Trading System</h1>
                <p className="text-xs text-muted-foreground">Deep Learning + RL Portfolio Optimization</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Stock Selector */}
              <Select.Root value={selectedStock} onValueChange={setSelectedStock}>
                <Select.Trigger className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-border bg-card hover:bg-secondary/50 transition-colors min-w-[200px]">
                  <TrendingUp className="h-4 w-4 text-primary" />
                  <Select.Value>
                    {selectedStock} • {currentStockInfo?.label}
                  </Select.Value>
                  <ChevronDown className="h-4 w-4 text-muted-foreground ml-auto" />
                </Select.Trigger>
                <Select.Portal>
                  <Select.Content className="overflow-hidden bg-card rounded-lg border border-border shadow-lg z-50">
                    <Select.Viewport className="p-1">
                      {AVAILABLE_STOCKS.map((stock) => (
                        <Select.Item
                          key={stock.value}
                          value={stock.value}
                          className="flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer hover:bg-secondary/50 outline-none data-[highlighted]:bg-secondary/50"
                        >
                          <Select.ItemText>
                            <span className="font-medium">{stock.value}</span>
                            <span className="text-muted-foreground ml-2 text-sm">{stock.label}</span>
                          </Select.ItemText>
                          <Select.ItemIndicator className="ml-auto">
                            <Check className="h-4 w-4 text-primary" />
                          </Select.ItemIndicator>
                        </Select.Item>
                      ))}
                    </Select.Viewport>
                  </Select.Content>
                </Select.Portal>
              </Select.Root>

              {/* Settings Button */}
              <button
                onClick={() => setConfigOpen(true)}
                className="p-2 rounded-lg border border-border hover:bg-secondary/50 transition-colors"
              >
                <Settings className="h-5 w-5" />
              </button>

              {/* Theme Toggle */}
              <button
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                className="p-2 rounded-lg border border-border hover:bg-secondary/50 transition-colors"
              >
                {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </button>

              {/* Run Button */}
              <button
                onClick={runPipeline}
                disabled={isRunning}
                className={cn(
                  'flex items-center gap-2 px-5 py-2 rounded-lg font-medium transition-all',
                  isRunning
                    ? 'bg-primary/50 text-primary-foreground cursor-not-allowed'
                    : 'bg-primary text-primary-foreground hover:bg-primary/90 shadow-lg shadow-primary/25'
                )}
              >
                {isRunning ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Run Pipeline
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Progress Bar */}
      <AnimatePresence>
        {isRunning && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-b border-border bg-card"
          >
            <div className="container mx-auto px-4 py-3">
              <div className="flex items-center gap-4">
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                <div className="flex-1">
                  <Progress.Root className="h-2 bg-secondary rounded-full overflow-hidden">
                    <Progress.Indicator
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </Progress.Root>
                </div>
                <span className="text-sm text-muted-foreground">{progress.toFixed(0)}%</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {/* Market Price Banner with Trading Panel */}
        {marketPrice && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 rounded-xl border border-border bg-card"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div>
                  <div className="text-sm text-muted-foreground">{selectedStock}</div>
                  <div className="text-2xl font-bold">{formatCurrency(marketPrice.price)}</div>
                </div>
                <div className={cn(
                  'flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium',
                  marketPrice.change >= 0 ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'
                )}>
                  {marketPrice.change >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                  {marketPrice.change >= 0 ? '+' : ''}{marketPrice.change.toFixed(2)} ({marketPrice.change_pct.toFixed(2)}%)
                </div>
                {/* Current holding badge */}
                {wallet?.portfolio[selectedStock] && (
                  <div className="px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium">
                    Holding: {wallet.portfolio[selectedStock].shares} shares
                    <span className={cn(
                      'ml-2',
                      wallet.portfolio[selectedStock].pnl >= 0 ? 'text-green-500' : 'text-red-500'
                    )}>
                      ({wallet.portfolio[selectedStock].pnl >= 0 ? '+' : ''}{formatCurrency(wallet.portfolio[selectedStock].pnl)})
                    </span>
                  </div>
                )}
              </div>
              
              {/* Trading Controls */}
              <div className="flex items-center gap-4">
                <div className="flex gap-2 text-sm">
                  <div>
                    <span className="text-muted-foreground">High: </span>
                    <span className="font-medium">{formatCurrency(marketPrice.high)}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Low: </span>
                    <span className="font-medium">{formatCurrency(marketPrice.low)}</span>
                  </div>
                </div>
                
                <div className="h-8 w-px bg-border" />
                
                {/* Quick Trade Panel */}
                <div className="flex items-center gap-2">
                  <div className="flex items-center border border-border rounded-lg">
                    <button
                      onClick={() => setTradeQuantity(Math.max(1, tradeQuantity - 1))}
                      className="p-2 hover:bg-secondary/50 rounded-l-lg"
                    >
                      <MinusCircle className="h-4 w-4" />
                    </button>
                    <input
                      type="number"
                      value={tradeQuantity}
                      onChange={(e) => setTradeQuantity(Math.max(1, parseInt(e.target.value) || 1))}
                      className="w-16 text-center bg-transparent border-x border-border py-1 text-sm"
                    />
                    <button
                      onClick={() => setTradeQuantity(tradeQuantity + 1)}
                      className="p-2 hover:bg-secondary/50 rounded-r-lg"
                    >
                      <PlusCircle className="h-4 w-4" />
                    </button>
                  </div>
                  
                  <span className="text-sm text-muted-foreground">
                    = {formatCurrency(tradeQuantity * marketPrice.price)}
                  </span>
                  
                  <button
                    onClick={() => executeTrade('buy')}
                    disabled={isTrading || (wallet?.balance || 0) < tradeQuantity * marketPrice.price}
                    className="flex items-center gap-1 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isTrading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ShoppingCart className="h-4 w-4" />}
                    Buy
                  </button>
                  
                  <button
                    onClick={() => executeTrade('sell')}
                    disabled={isTrading || !wallet?.portfolio[selectedStock]}
                    className="flex items-center gap-1 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isTrading ? <Loader2 className="h-4 w-4 animate-spin" /> : <TrendingDown className="h-4 w-4" />}
                    Sell
                  </button>
                  
                  <button
                    onClick={autoTrade}
                    disabled={isTrading}
                    className="flex items-center gap-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 transition-colors"
                    title="Let AI decide based on model prediction"
                  >
                    {isTrading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Brain className="h-4 w-4" />}
                    AI Trade
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Portfolio Holdings */}
        {wallet && Object.keys(wallet.portfolio).length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 rounded-xl border border-border bg-card"
          >
            <div className="flex items-center gap-2 mb-3">
              <Briefcase className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Your Holdings</h3>
              <span className="text-sm text-muted-foreground">({wallet.holdings_count} stocks)</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {Object.entries(wallet.portfolio).map(([symbol, holding]) => (
                <div
                  key={symbol}
                  onClick={() => setSelectedStock(symbol)}
                  className={cn(
                    'p-3 rounded-lg border cursor-pointer transition-all hover:shadow-md',
                    selectedStock === symbol ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50'
                  )}
                >
                  <div className="font-medium">{symbol}</div>
                  <div className="text-sm text-muted-foreground">{holding.shares} shares</div>
                  <div className="text-sm font-medium">{formatCurrency(holding.current_value)}</div>
                  <div className={cn(
                    'text-xs',
                    holding.pnl >= 0 ? 'text-green-500' : 'text-red-500'
                  )}>
                    {holding.pnl >= 0 ? '+' : ''}{formatCurrency(holding.pnl)} ({holding.pnl_pct.toFixed(1)}%)
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
          <StatCard
            icon={Target}
            label="Win Rate"
            value={formatPercent(result?.win_rate)}
            trend={(result?.win_rate || 0) > 0.5 ? 'up' : 'neutral'}
            color="green"
          />
          <StatCard
            icon={DollarSign}
            label="Total Return"
            value={formatPercent(result?.total_return)}
            trend={(result?.total_return || 0) > 0 ? 'up' : (result?.total_return || 0) < 0 ? 'down' : 'neutral'}
            color={(result?.total_return || 0) >= 0 ? 'green' : 'red'}
          />
          <StatCard
            icon={Activity}
            label="Sharpe Ratio"
            value={safeToFixed(result?.sharpe_ratio)}
            trend={(result?.sharpe_ratio || 0) > 1 ? 'up' : 'neutral'}
            color="primary"
          />
          <StatCard
            icon={Shield}
            label="Profit Factor"
            value={safeToFixed(result?.profit_factor)}
            trend={(result?.profit_factor || 0) > 1.5 ? 'up' : 'neutral'}
            color="yellow"
          />
          <StatCard
            icon={AlertCircle}
            label="Max Drawdown"
            value={formatPercent(result?.max_drawdown)}
            trend="down"
            color="red"
          />
          <StatCard
            icon={BarChart3}
            label="Total Trades"
            value={String(result?.total_trades || 0)}
            color="primary"
          />
        </div>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Equity Curve */}
          <div className="lg:col-span-2 rounded-xl border border-border bg-card p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <LineChart className="h-5 w-5 text-primary" />
                <h2 className="text-lg font-semibold">Equity Curve</h2>
              </div>
              <span className="text-sm text-muted-foreground">
                {result?.trades?.length || 0} trades executed
              </span>
            </div>
            <div className="h-80">
              {result?.equity_curve && result.equity_curve.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={result.equity_curve}>
                    <defs>
                      <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.5} />
                    <XAxis
                      dataKey="date"
                      stroke="hsl(var(--muted-foreground))"
                      tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                    />
                    <YAxis
                      stroke="hsl(var(--muted-foreground))"
                      tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                      tickFormatter={(val) => `₹${(val / 1000).toFixed(0)}K`}
                    />
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--popover))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px',
                      }}
                      formatter={(val: number) => [formatCurrency(val), 'Equity']}
                    />
                    <Area
                      type="monotone"
                      dataKey="equity"
                      stroke="hsl(var(--primary))"
                      fill="url(#equityGradient)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-muted-foreground">
                  <div className="text-center">
                    <LineChart className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p>Run pipeline to see equity curve</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Sentiment Panel */}
          <div className="rounded-xl border border-border bg-card p-5">
            <div className="flex items-center gap-2 mb-4">
              <Zap className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Sentiment Analysis</h2>
            </div>
            {sentiment ? (
              <div className="space-y-4">
                <div className="text-center py-4">
                  <div className={cn(
                    'text-5xl font-bold',
                    sentiment.sentiment.overall_sentiment > 0.1 ? 'text-green-500' :
                    sentiment.sentiment.overall_sentiment < -0.1 ? 'text-red-500' : 'text-yellow-500'
                  )}>
                    {(sentiment.sentiment.overall_sentiment * 100).toFixed(0)}
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">Sentiment Score</div>
                  <div className={cn(
                    'inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium mt-2',
                    sentiment.sentiment.sentiment_label === 'Bullish' ? 'bg-green-500/10 text-green-500' :
                    sentiment.sentiment.sentiment_label === 'Bearish' ? 'bg-red-500/10 text-red-500' :
                    'bg-yellow-500/10 text-yellow-500'
                  )}>
                    {sentiment.sentiment.sentiment_label === 'Bullish' && <TrendingUp className="h-4 w-4" />}
                    {sentiment.sentiment.sentiment_label === 'Bearish' && <TrendingDown className="h-4 w-4" />}
                    {sentiment.sentiment.sentiment_label}
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Positive</span>
                      <span className="text-green-500">{(sentiment.sentiment.positive_ratio * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-green-500 rounded-full transition-all"
                        style={{ width: `${sentiment.sentiment.positive_ratio * 100}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Neutral</span>
                      <span className="text-yellow-500">{(sentiment.sentiment.neutral_ratio * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-yellow-500 rounded-full transition-all"
                        style={{ width: `${sentiment.sentiment.neutral_ratio * 100}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Negative</span>
                      <span className="text-red-500">{(sentiment.sentiment.negative_ratio * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-red-500 rounded-full transition-all"
                        style={{ width: `${sentiment.sentiment.negative_ratio * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="flex justify-between text-sm text-muted-foreground pt-2 border-t border-border">
                  <span>{sentiment.sentiment.news_volume} articles analyzed</span>
                  <span>Live data</span>
                </div>
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <Brain className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>Loading sentiment...</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Trade History */}
        <div className="mt-6 rounded-xl border border-border bg-card p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Briefcase className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Trade History</h2>
            </div>
            {result?.trades && result.trades.length > 10 && (
              <span className="text-sm text-muted-foreground">
                Showing 10 of {result.trades.length} trades
              </span>
            )}
          </div>
          <TradeHistory trades={result?.trades || []} />
        </div>

        {/* Model Info */}
        <div className="mt-6 grid md:grid-cols-3 gap-4">
          <div className="rounded-xl border border-border bg-card p-4">
            <div className="flex items-center gap-2 mb-2">
              <Cpu className="h-4 w-4 text-primary" />
              <span className="font-medium">Current Model</span>
            </div>
            <div className="text-2xl font-bold capitalize">{config.model_type}</div>
            <div className="text-sm text-muted-foreground">Deep Learning Engine</div>
          </div>
          <div className="rounded-xl border border-border bg-card p-4">
            <div className="flex items-center gap-2 mb-2">
              <Layers className="h-4 w-4 text-primary" />
              <span className="font-medium">Features</span>
            </div>
            <div className="flex flex-wrap gap-1 mt-1">
              {config.use_technical && <span className="px-2 py-0.5 bg-primary/10 text-primary rounded text-xs">Technical</span>}
              {config.use_sentiment && <span className="px-2 py-0.5 bg-green-500/10 text-green-500 rounded text-xs">Sentiment</span>}
              {config.use_volume && <span className="px-2 py-0.5 bg-yellow-500/10 text-yellow-500 rounded text-xs">Volume</span>}
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card p-4">
            <div className="flex items-center gap-2 mb-2">
              <Shield className="h-4 w-4 text-primary" />
              <span className="font-medium">Risk Settings</span>
            </div>
            <div className="text-sm space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Stop Loss</span>
                <span className="text-red-500">{(config.stop_loss_pct * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Take Profit</span>
                <span className="text-green-500">{(config.take_profit_pct * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Config Dialog */}
      <Dialog.Root open={configOpen} onOpenChange={setConfigOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/50 z-50" />
          <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md max-h-[90vh] overflow-y-auto bg-card rounded-xl border border-border shadow-xl z-50 p-6">
            <div className="flex items-center justify-between mb-6">
              <Dialog.Title className="text-lg font-semibold">Configuration</Dialog.Title>
              <Dialog.Close asChild>
                <button className="p-1 rounded hover:bg-secondary">
                  <X className="h-5 w-5" />
                </button>
              </Dialog.Close>
            </div>
            <ConfigPanel config={config} setConfig={setConfig} onApply={applyConfig} />
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      {/* Wallet Dialog */}
      <Dialog.Root open={walletOpen} onOpenChange={setWalletOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/50 z-50" />
          <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-card rounded-xl border border-border shadow-xl z-50 p-6">
            <div className="flex items-center justify-between mb-6">
              <Dialog.Title className="text-lg font-semibold flex items-center gap-2">
                <Wallet className="h-5 w-5 text-primary" />
                Wallet & Transactions
              </Dialog.Title>
              <Dialog.Close asChild>
                <button className="p-1 rounded hover:bg-secondary">
                  <X className="h-5 w-5" />
                </button>
              </Dialog.Close>
            </div>
            
            {wallet && (
              <div className="space-y-6">
                {/* Wallet Summary */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-3 rounded-lg bg-secondary/30">
                    <div className="text-sm text-muted-foreground">Cash Balance</div>
                    <div className="text-xl font-bold">{formatCurrency(wallet.balance)}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-secondary/30">
                    <div className="text-sm text-muted-foreground">Portfolio Value</div>
                    <div className="text-xl font-bold">{formatCurrency(wallet.portfolio_value)}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-secondary/30">
                    <div className="text-sm text-muted-foreground">Realized P&L</div>
                    <div className={cn('text-xl font-bold', wallet.realized_pnl >= 0 ? 'text-green-500' : 'text-red-500')}>
                      {wallet.realized_pnl >= 0 ? '+' : ''}{formatCurrency(wallet.realized_pnl)}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-secondary/30">
                    <div className="text-sm text-muted-foreground">Unrealized P&L</div>
                    <div className={cn('text-xl font-bold', wallet.unrealized_pnl >= 0 ? 'text-green-500' : 'text-red-500')}>
                      {wallet.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(wallet.unrealized_pnl)}
                    </div>
                  </div>
                </div>

                {/* Holdings */}
                {Object.keys(wallet.portfolio).length > 0 && (
                  <div>
                    <h3 className="font-medium mb-3">Current Holdings</h3>
                    <div className="space-y-2">
                      {Object.entries(wallet.portfolio).map(([symbol, holding]) => (
                        <div key={symbol} className="flex items-center justify-between p-3 rounded-lg bg-secondary/20">
                          <div>
                            <div className="font-medium">{symbol}</div>
                            <div className="text-sm text-muted-foreground">
                              {holding.shares} shares @ {formatCurrency(holding.avg_price)}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-medium">{formatCurrency(holding.current_value)}</div>
                            <div className={cn('text-sm', holding.pnl >= 0 ? 'text-green-500' : 'text-red-500')}>
                              {holding.pnl >= 0 ? '+' : ''}{formatCurrency(holding.pnl)} ({holding.pnl_pct.toFixed(2)}%)
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Transaction History */}
                <div>
                  <h3 className="font-medium mb-3">Recent Transactions</h3>
                  {transactions.length > 0 ? (
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {transactions.map((tx, idx) => (
                        <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-secondary/20">
                          <div className="flex items-center gap-3">
                            <div className={cn(
                              'w-8 h-8 rounded-full flex items-center justify-center',
                              tx.type === 'BUY' ? 'bg-green-500/20' : 'bg-red-500/20'
                            )}>
                              {tx.type === 'BUY' ? (
                                <ShoppingCart className="h-4 w-4 text-green-500" />
                              ) : (
                                <TrendingDown className="h-4 w-4 text-red-500" />
                              )}
                            </div>
                            <div>
                              <div className="font-medium">{tx.type} {tx.symbol}</div>
                              <div className="text-sm text-muted-foreground">
                                {tx.quantity} shares @ {formatCurrency(tx.price)}
                              </div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className={cn('font-medium', tx.type === 'BUY' ? 'text-red-500' : 'text-green-500')}>
                              {tx.type === 'BUY' ? '-' : '+'}{formatCurrency(tx.total)}
                            </div>
                            {tx.pnl !== undefined && (
                              <div className={cn('text-sm', tx.pnl >= 0 ? 'text-green-500' : 'text-red-500')}>
                                P&L: {tx.pnl >= 0 ? '+' : ''}{formatCurrency(tx.pnl)}
                              </div>
                            )}
                            <div className="text-xs text-muted-foreground">
                              {new Date(tx.timestamp).toLocaleString()}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      No transactions yet. Start trading!
                    </div>
                  )}
                </div>

                <button
                  onClick={() => {
                    resetWallet();
                    setWalletOpen(false);
                  }}
                  className="w-full py-2 px-4 border border-red-500 text-red-500 rounded-lg hover:bg-red-500/10 transition-colors"
                >
                  Reset Wallet to ₹1,00,000
                </button>
              </div>
            )}
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      {/* Error Toast */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed bottom-4 right-4 bg-red-500 text-white px-4 py-3 rounded-lg shadow-lg flex items-center gap-2"
          >
            <AlertCircle className="h-5 w-5" />
            {error}
            <button onClick={() => setError(null)} className="ml-2 hover:opacity-80">
              <X className="h-4 w-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

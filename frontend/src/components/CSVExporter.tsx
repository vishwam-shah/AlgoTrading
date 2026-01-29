/**
 * CSV Exporter Component
 * One-click export of all pipeline results
 */

import React, { useState } from 'react';
import { Download, FileText, TrendingUp, BarChart3, Brain } from 'lucide-react';

interface CSVExporterProps {
    pipelineId: string;
    backtestResults: Record<string, any>;
    signals: Record<string, any>;
    pipelineData?: any;
}

export const CSVExporter: React.FC<CSVExporterProps> = ({
    pipelineId,
    backtestResults,
    signals,
    pipelineData
}) => {
    const [isOpen, setIsOpen] = useState(false);

    const formatCSVValue = (value: any): string => {
        if (value === null || value === undefined) return '';
        if (typeof value === 'number') {
            return value.toFixed(4);
        }
        return String(value).replace(/,/g, ';'); // Replace commas to avoid CSV issues
    };

    const generateCSVHeader = () => {
        const now = new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' });
        return `# Pipeline Run: ${pipelineId}
# Generated: ${now}
# Symbols: ${Object.keys(backtestResults).join(', ')}
# Data Range: 2022-01-01 to ${new Date().toISOString().split('T')[0]}

`;
    };

    const exportPerStockMetrics = () => {
        let csv = generateCSVHeader();
        csv += 'Symbol,XGB Accuracy,LGB Accuracy,Ensemble Accuracy,Total Return,Sharpe Ratio,Win Rate,Total Trades,Signal,Confidence,Expected Return\n';

        Object.entries(backtestResults).forEach(([symbol, data]: [string, any]) => {
            const signal = signals[symbol] || {};
            const modelPreds = data.model_predictions || {};

            csv += [
                symbol,
                modelPreds.xgb_accuracy ? (modelPreds.xgb_accuracy * 100).toFixed(1) + '%' : 'N/A',
                modelPreds.lgb_accuracy ? (modelPreds.lgb_accuracy * 100).toFixed(1) + '%' : 'N/A',
                modelPreds.ensemble_accuracy ? (modelPreds.ensemble_accuracy * 100).toFixed(1) + '%' : 'N/A',
                data.total_return ? (data.total_return * 100).toFixed(2) + '%' : 'N/A',
                data.sharpe_ratio ? data.sharpe_ratio.toFixed(2) : 'N/A',
                data.win_rate ? (data.win_rate * 100).toFixed(1) + '%' : 'N/A',
                data.total_trades || 0,
                signal.action || 'N/A',
                signal.confidence ? (signal.confidence * 100).toFixed(1) + '%' : 'N/A',
                signal.expected_return ? (signal.expected_return * 100).toFixed(2) + '%' : 'N/A'
            ].join(',') + '\n';
        });

        downloadCSV(csv, `per_stock_metrics_${pipelineId}.csv`);
    };

    const exportTradingSignals = () => {
        let csv = generateCSVHeader();
        csv += 'Symbol,Signal,Confidence,Direction Probability,Expected Return,Current Price,Recommendation\n';

        Object.entries(signals).forEach(([symbol, signal]: [string, any]) => {
            const recommendation = signal.action === 'BUY' ? 'Execute Trade' :
                signal.action === 'SELL' ? 'Execute Trade' :
                    'Wait for better signal';

            csv += [
                symbol,
                signal.action || 'N/A',
                signal.confidence ? (signal.confidence * 100).toFixed(1) + '%' : 'N/A',
                signal.direction_probability ? signal.direction_probability.toFixed(3) : 'N/A',
                signal.expected_return ? (signal.expected_return * 100).toFixed(2) + '%' : 'N/A',
                signal.current_price ? '₹' + signal.current_price.toFixed(2) : 'N/A',
                recommendation
            ].join(',') + '\n';
        });

        downloadCSV(csv, `trading_signals_${pipelineId}.csv`);
    };

    const exportBacktestTrades = () => {
        let csv = generateCSVHeader();
        csv += 'Symbol,Entry Date,Exit Date,Direction,Entry Price,Exit Price,Shares,P&L,Return %,Exit Reason\n';

        Object.entries(backtestResults).forEach(([symbol, data]: [string, any]) => {
            if (data.trades && Array.isArray(data.trades)) {
                data.trades.forEach((trade: any) => {
                    csv += [
                        symbol,
                        trade.entry_date || 'N/A',
                        trade.exit_date || 'N/A',
                        trade.direction === 1 ? 'LONG' : 'SHORT',
                        trade.entry_price ? '₹' + trade.entry_price.toFixed(2) : 'N/A',
                        trade.exit_price ? '₹' + trade.exit_price.toFixed(2) : 'N/A',
                        trade.shares || 0,
                        trade.pnl ? '₹' + trade.pnl.toFixed(2) : 'N/A',
                        trade.return_pct ? trade.return_pct.toFixed(2) + '%' : 'N/A',
                        trade.exit_reason || 'N/A'
                    ].join(',') + '\n';
                });
            }
        });

        downloadCSV(csv, `backtest_trades_${pipelineId}.csv`);
    };

    const exportFeatureImportance = () => {
        let csv = generateCSVHeader();
        csv += 'Symbol,Rank,Feature Name,Importance Score\n';

        Object.entries(backtestResults).forEach(([symbol, data]: [string, any]) => {
            if (data.feature_importance && Array.isArray(data.feature_importance)) {
                data.feature_importance.forEach((feat: any, idx: number) => {
                    csv += [
                        symbol,
                        idx + 1,
                        feat.feature || 'N/A',
                        feat.importance ? feat.importance.toFixed(6) : 'N/A'
                    ].join(',') + '\n';
                });
            }
        });

        downloadCSV(csv, `feature_importance_${pipelineId}.csv`);
    };

    const exportAllResults = () => {
        let csv = `# COMPREHENSIVE PIPELINE RESULTS
# Pipeline Run: ${pipelineId}
# Generated: ${new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })}
# Symbols: ${Object.keys(backtestResults).join(', ')}

==================== PER-STOCK METRICS ====================
Symbol,XGB Accuracy,LGB Accuracy,Ensemble Accuracy,Total Return,Sharpe Ratio,Win Rate,Total Trades
`;

        Object.entries(backtestResults).forEach(([symbol, data]: [string, any]) => {
            const modelPreds = data.model_predictions || {};
            csv += [
                symbol,
                modelPreds.xgb_accuracy ? (modelPreds.xgb_accuracy * 100).toFixed(1) + '%' : 'N/A',
                modelPreds.lgb_accuracy ? (modelPreds.lgb_accuracy * 100).toFixed(1) + '%' : 'N/A',
                modelPreds.ensemble_accuracy ? (modelPreds.ensemble_accuracy * 100).toFixed(1) + '%' : 'N/A',
                data.total_return ? (data.total_return * 100).toFixed(2) + '%' : 'N/A',
                data.sharpe_ratio ? data.sharpe_ratio.toFixed(2) : 'N/A',
                data.win_rate ? (data.win_rate * 100).toFixed(1) + '%' : 'N/A',
                data.total_trades || 0
            ].join(',') + '\n';
        });

        csv += `\n==================== TRADING SIGNALS ====================
Symbol,Signal,Confidence,Expected Return,Current Price
`;

        Object.entries(signals).forEach(([symbol, signal]: [string, any]) => {
            csv += [
                symbol,
                signal.action || 'N/A',
                signal.confidence ? (signal.confidence * 100).toFixed(1) + '%' : 'N/A',
                signal.expected_return ? (signal.expected_return * 100).toFixed(2) + '%' : 'N/A',
                signal.current_price ? '₹' + signal.current_price.toFixed(2) : 'N/A'
            ].join(',') + '\n';
        });

        downloadCSV(csv, `all_results_${pipelineId}.csv`);
    };

    const downloadCSV = (content: string, filename: string) => {
        const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        setIsOpen(false);
    };

    const exportOptions = [
        {
            id: 'metrics',
            label: 'Per-Stock Metrics',
            description: 'XGB/LGB/Ensemble accuracy, returns, Sharpe ratio',
            icon: BarChart3,
            action: exportPerStockMetrics
        },
        {
            id: 'signals',
            label: 'Trading Signals',
            description: 'Action, confidence, expected return, prices',
            icon: TrendingUp,
            action: exportTradingSignals
        },
        {
            id: 'trades',
            label: 'Backtest Trades',
            description: 'All trades with entry/exit data and P&L',
            icon: FileText,
            action: exportBacktestTrades
        },
        {
            id: 'features',
            label: 'Feature Importance',
            description: 'Top predictive features per stock',
            icon: Brain,
            action: exportFeatureImportance
        }
    ];

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
            >
                <Download className="w-4 h-4" />
                Export Results
            </button>

            {isOpen && (
                <>
                    <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
                    <div className="absolute right-0 mt-2 w-80 bg-secondary border border-border rounded-lg shadow-xl z-50 overflow-hidden">
                        <div className="p-4 border-b border-border">
                            <h3 className="font-semibold text-sm">Export Pipeline Results</h3>
                            <p className="text-xs text-muted-foreground mt-1">Choose what to export as CSV</p>
                        </div>

                        <div className="p-2">
                            {exportOptions.map(option => {
                                const Icon = option.icon;
                                return (
                                    <button
                                        key={option.id}
                                        onClick={option.action}
                                        className="w-full flex items-start gap-3 p-3 rounded-md hover:bg-background transition-colors text-left"
                                    >
                                        <Icon className="w-5 h-5 mt-0.5 text-primary flex-shrink-0" />
                                        <div className="flex-1 min-w-0">
                                            <div className="font-medium text-sm">{option.label}</div>
                                            <div className="text-xs text-muted-foreground mt-0.5">{option.description}</div>
                                        </div>
                                    </button>
                                );
                            })}

                            <div className="border-t border-border my-2" />

                            <button
                                onClick={exportAllResults}
                                className="w-full flex items-start gap-3 p-3 rounded-md bg-primary/10 hover:bg-primary/20 transition-colors text-left"
                            >
                                <Download className="w-5 h-5 mt-0.5 text-primary flex-shrink-0" />
                                <div className="flex-1 min-w-0">
                                    <div className="font-semibold text-sm text-primary">Export All Results</div>
                                    <div className="text-xs text-muted-foreground mt-0.5">Comprehensive export with all data</div>
                                </div>
                            </button>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

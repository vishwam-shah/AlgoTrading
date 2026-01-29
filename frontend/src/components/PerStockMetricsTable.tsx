/**
 * Per-Stock Metrics Display Component
 * Shows individual metrics for each stock in the pipeline run
 */

import React from 'react';

interface StockMetrics {
    symbol: string;
    directionalAccuracy?: number;
    totalReturn?: number;
    sharpeRatio?: number;
    winRate?: number;
    totalTrades?: number;
    signalAction?: string;
    signalConfidence?: number;
    expectedReturn?: number;
    modelPredictions?: {
        xgb_accuracy?: number;
        lgb_accuracy?: number;
        ensemble_accuracy?: number;
    };
}

interface PerStockMetricsProps {
    backtestResults: Record<string, any>;
    signals: Record<string, any>;
}

export const PerStockMetricsTable: React.FC<PerStockMetricsProps> = ({ backtestResults, signals }) => {
    // Combine backtest and signal data
    const allSymbols = Array.from(
        new Set([...Object.keys(backtestResults || {}), ...Object.keys(signals || {})])
    );

    if (allSymbols.length === 0) {
        return (
            <div className="text-center py-8 text-muted-foreground">
                <p className="text-sm">No per-stock metrics available. Run a pipeline to see results.</p>
            </div>
        );
    }

    const metrics: StockMetrics[] = allSymbols.map(symbol => {
        const backtest = backtestResults?.[symbol] || {};
        const signal = signals?.[symbol] || {};

        // Access model_predictions correctly
        const modelPreds = backtest.model_predictions || {};

        return {
            symbol,
            directionalAccuracy: backtest.directional_accuracy,
            totalReturn: backtest.total_return,
            sharpeRatio: backtest.sharpe_ratio,
            winRate: backtest.win_rate,
            totalTrades: backtest.total_trades,
            signalAction: signal.action,
            signalConfidence: signal.confidence,
            expectedReturn: signal.expected_return,
            // Add model predictions to metrics
            modelPredictions: modelPreds
        };
    });

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-border">
                        <th className="text-left py-3 px-4 font-medium">Symbol</th>
                        <th className="text-left py-3 px-4 font-medium">XGB Acc.</th>
                        <th className="text-left py-3 px-4 font-medium">LGB Acc.</th>
                        <th className="text-left py-3 px-4 font-medium">Ensemble Acc.</th>
                        <th className="text-left py-3 px-4 font-medium">Total Return</th>
                        <th className="text-left py-3 px-4 font-medium">Sharpe</th>
                        <th className="text-left py-3 px-4 font-medium">Win Rate</th>
                        <th className="text-left py-3 px-4 font-medium">Trades</th>
                        <th className="text-left py-3 px-4 font-medium">Signal</th>
                        <th className="text-left py-3 px-4 font-medium">Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {metrics.map((m, idx) => (
                        <tr key={m.symbol} className={`border-b border-border/30 hover:bg-secondary/20 ${idx % 2 === 0 ? 'bg-secondary/5' : ''}`}>
                            <td className="py-3 px-4 font-mono font-medium">{m.symbol}</td>

                            {/* XGBoost Accuracy */}
                            <td className="py-3 px-4">
                                {m.modelPredictions?.xgb_accuracy != null ? (
                                    <span className={m.modelPredictions.xgb_accuracy >= 0.55 ? 'text-green-500 font-medium' : m.modelPredictions.xgb_accuracy < 0.50 ? 'text-red-500' : 'text-yellow-600'}>
                                        {(m.modelPredictions.xgb_accuracy * 100).toFixed(1)}%
                                    </span>
                                ) : (
                                    <span className="text-muted-foreground text-xs">N/A</span>
                                )}
                            </td>

                            {/* LightGBM Accuracy */}
                            <td className="py-3 px-4">
                                {m.modelPredictions?.lgb_accuracy != null ? (
                                    <span className={m.modelPredictions.lgb_accuracy >= 0.55 ? 'text-green-500 font-medium' : m.modelPredictions.lgb_accuracy < 0.50 ? 'text-red-500' : 'text-yellow-600'}>
                                        {(m.modelPredictions.lgb_accuracy * 100).toFixed(1)}%
                                    </span>
                                ) : (
                                    <span className="text-muted-foreground text-xs">N/A</span>
                                )}
                            </td>

                            {/* Ensemble Accuracy (Bold) */}
                            <td className="py-3 px-4">
                                {m.modelPredictions?.ensemble_accuracy != null ? (
                                    <span className={`font-bold ${m.modelPredictions.ensemble_accuracy >= 0.55 ? 'text-green-500' : m.modelPredictions.ensemble_accuracy < 0.50 ? 'text-red-500' : 'text-yellow-600'}`}>
                                        {(m.modelPredictions.ensemble_accuracy * 100).toFixed(1)}%
                                    </span>
                                ) : m.directionalAccuracy != null ? (
                                    <span className={`font-bold ${m.directionalAccuracy >= 0.55 ? 'text-green-500' : m.directionalAccuracy < 0.50 ? 'text-red-500' : 'text-yellow-600'}`}>
                                        {(m.directionalAccuracy * 100).toFixed(1)}%
                                    </span>
                                ) : (
                                    <span className="text-muted-foreground text-xs">N/A</span>
                                )}
                            </td>

                            <td className="py-3 px-4">
                                {m.totalReturn != null ? (
                                    <span className={m.totalReturn >= 0 ? 'text-green-500' : 'text-red-500'}>
                                        {(m.totalReturn * 100).toFixed(2)}%
                                    </span>
                                ) : (
                                    <span className="text-muted-foreground text-xs">N/A</span>
                                )}
                            </td>
                            <td className="py-3 px-4">
                                {m.sharpeRatio != null ? m.sharpeRatio.toFixed(2) : 'N/A'}
                            </td>
                            <td className="py-3 px-4">
                                {m.winRate != null ? `${(m.winRate * 100).toFixed(1)}%` : 'N/A'}
                            </td>
                            <td className="py-3 px-4">{m.totalTrades || 0}</td>
                            <td className="py-3 px-4">
                                {m.signalAction ? (
                                    <span
                                        className={`px-2 py-0.5 rounded text-xs font-medium ${m.signalAction === 'BUY'
                                            ? 'bg-green-500/20 text-green-600'
                                            : m.signalAction === 'SELL'
                                                ? 'bg-red-500/20 text-red-600'
                                                : 'bg-yellow-500/20 text-yellow-600'
                                            }`}
                                    >
                                        {m.signalAction}
                                    </span>
                                ) : (
                                    <span className="text-muted-foreground">—</span>
                                )}
                            </td>
                            <td className="py-3 px-4">
                                {m.signalConfidence != null ? (
                                    <span className={m.signalConfidence >= 0.65 ? 'text-green-500 font-medium' : m.signalConfidence <= 0.55 ? 'text-yellow-600' : ''}>
                                        {(m.signalConfidence * 100).toFixed(1)}%
                                    </span>
                                ) : (
                                    <span className="text-muted-foreground">—</span>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
            <div className="mt-4 p-3 rounded-lg bg-blue-500/10 border border-blue-500/30">
                <p className="text-xs text-blue-600">
                    <strong>Note:</strong> Directional Accuracy shows how well each model (XGBoost, LightGBM, Ensemble) predicts price direction (up/down) for each stock.
                    Confidence below 55% indicates the model should HOLD (uncertain). Target: 65-70% accuracy.
                </p>
            </div>
        </div>
    );
};

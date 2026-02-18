/**
 * Per-Stock Metrics Display Component
 * Shows individual metrics for each stock in the pipeline run
 * Auto-detects V3 data and shows V3-specific columns when present
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
    // Phase 1: Strategy Optimization Metrics
    qualityScore?: number;
    entryApproved?: boolean;
    positionSizeUsd?: number;
    riskPct?: number;
    entryReasons?: string;
    // V3 extras
    v3BestWindow?: string;
    v3AllWindows?: Record<string, any>;
    v3NFeatures?: number;
    v3WinRateLong?: number;
    v3WinRateShort?: number;
    v3ProfitFactor?: number;
}

interface PerStockMetricsProps {
    backtestResults: Record<string, any>;
    signals: Record<string, any>;
}

function AccuracyCell({ value }: { value?: number }) {
    if (value == null) return <span className="text-muted-foreground text-xs">N/A</span>;
    const color = value >= 0.55 ? 'text-green-500 font-medium' : value < 0.50 ? 'text-red-500' : 'text-yellow-600';
    return <span className={color}>{(value * 100).toFixed(1)}%</span>;
}

export const PerStockMetricsTable: React.FC<PerStockMetricsProps> = ({ backtestResults, signals }) => {
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

    // Auto-detect V3 mode
    const isV3 = Object.values(backtestResults || {}).some((r: any) => r.v3_all_windows);

    const metrics: StockMetrics[] = allSymbols.map(symbol => {
        const backtest = backtestResults?.[symbol] || {};
        const signal = signals?.[symbol] || {};
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
            modelPredictions: modelPreds,
            qualityScore: signal.quality_score,
            entryApproved: signal.entry_approved,
            positionSizeUsd: signal.position_size_usd,
            riskPct: signal.risk_pct,
            entryReasons: signal.entry_reasons,
            // V3 extras
            v3BestWindow: backtest.v3_best_window,
            v3AllWindows: backtest.v3_all_windows,
            v3NFeatures: backtest.v3_n_features,
            v3WinRateLong: backtest.v3_all_windows?.[backtest.v3_best_window]?.win_rate_long,
            v3WinRateShort: backtest.v3_all_windows?.[backtest.v3_best_window]?.win_rate_short,
            v3ProfitFactor: backtest.profit_factor,
        };
    });

    if (isV3) {
        return (
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-border">
                            <th className="text-left py-3 px-4 font-medium">Symbol</th>
                            <th className="text-left py-3 px-4 font-medium">Best Window</th>
                            <th className="text-left py-3 px-4 font-medium">XGB Acc.</th>
                            <th className="text-left py-3 px-4 font-medium">LGB Acc.</th>
                            <th className="text-left py-3 px-4 font-medium">Ensemble Acc.</th>
                            <th className="text-left py-3 px-4 font-medium">Sharpe</th>
                            <th className="text-left py-3 px-4 font-medium">WR Long</th>
                            <th className="text-left py-3 px-4 font-medium">WR Short</th>
                            <th className="text-left py-3 px-4 font-medium">Profit Factor</th>
                            <th className="text-left py-3 px-4 font-medium">Val Days</th>
                            <th className="text-left py-3 px-4 font-medium">Features</th>
                            <th className="text-left py-3 px-4 font-medium">Signal</th>
                            <th className="text-left py-3 px-4 font-medium">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {metrics.map((m, idx) => (
                            <tr key={m.symbol} className={`border-b border-border/30 hover:bg-secondary/20 ${idx % 2 === 0 ? 'bg-secondary/5' : ''}`}>
                                <td className="py-3 px-4 font-mono font-medium">{m.symbol}</td>

                                <td className="py-3 px-4">
                                    <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-600 text-xs font-mono">
                                        {m.v3BestWindow || 'N/A'}
                                    </span>
                                </td>

                                <td className="py-3 px-4">
                                    <AccuracyCell value={m.modelPredictions?.xgb_accuracy} />
                                </td>

                                <td className="py-3 px-4">
                                    <AccuracyCell value={m.modelPredictions?.lgb_accuracy} />
                                </td>

                                <td className="py-3 px-4">
                                    {m.modelPredictions?.ensemble_accuracy != null ? (
                                        <span className={`font-bold ${m.modelPredictions.ensemble_accuracy >= 0.55 ? 'text-green-500' : m.modelPredictions.ensemble_accuracy < 0.50 ? 'text-red-500' : 'text-yellow-600'}`}>
                                            {(m.modelPredictions.ensemble_accuracy * 100).toFixed(1)}%
                                        </span>
                                    ) : (
                                        <span className="text-muted-foreground text-xs">N/A</span>
                                    )}
                                </td>

                                <td className="py-3 px-4">
                                    {m.sharpeRatio != null ? (
                                        <span className={m.sharpeRatio >= 1.0 ? 'text-green-500 font-medium' : m.sharpeRatio < 0 ? 'text-red-500' : ''}>
                                            {m.sharpeRatio.toFixed(2)}
                                        </span>
                                    ) : 'N/A'}
                                </td>

                                <td className="py-3 px-4">
                                    {m.v3WinRateLong != null ? (
                                        <span className={m.v3WinRateLong >= 0.55 ? 'text-green-500 font-medium' : m.v3WinRateLong < 0.50 ? 'text-red-500' : 'text-yellow-600'}>
                                            {(m.v3WinRateLong * 100).toFixed(1)}%
                                        </span>
                                    ) : <span className="text-muted-foreground text-xs">N/A</span>}
                                </td>

                                <td className="py-3 px-4">
                                    {m.v3WinRateShort != null ? (
                                        <span className={m.v3WinRateShort >= 0.55 ? 'text-green-500 font-medium' : m.v3WinRateShort < 0.50 ? 'text-red-500' : 'text-yellow-600'}>
                                            {(m.v3WinRateShort * 100).toFixed(1)}%
                                        </span>
                                    ) : <span className="text-muted-foreground text-xs">N/A</span>}
                                </td>

                                <td className="py-3 px-4">
                                    {m.v3ProfitFactor != null ? (
                                        <span className={m.v3ProfitFactor >= 1.5 ? 'text-green-500 font-medium' : m.v3ProfitFactor < 1.0 ? 'text-red-500' : 'text-yellow-600'}>
                                            {m.v3ProfitFactor.toFixed(2)}
                                        </span>
                                    ) : <span className="text-muted-foreground text-xs">N/A</span>}
                                </td>

                                <td className="py-3 px-4">{m.totalTrades || 0}</td>

                                <td className="py-3 px-4">
                                    <span className="text-xs font-mono">{m.v3NFeatures || 'N/A'}</span>
                                </td>

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

                {/* V3 All Windows Detail */}
                {metrics.some(m => m.v3AllWindows) && (
                    <details className="mt-4">
                        <summary className="cursor-pointer text-sm font-medium text-muted-foreground hover:text-foreground">
                            Show all window configs per stock
                        </summary>
                        <div className="mt-2 space-y-3">
                            {metrics.filter(m => m.v3AllWindows).map(m => (
                                <div key={m.symbol} className="p-3 rounded-lg bg-secondary/20 border border-border/50">
                                    <h4 className="text-sm font-mono font-medium mb-2">{m.symbol}</h4>
                                    <table className="w-full text-xs">
                                        <thead>
                                            <tr className="border-b border-border/50">
                                                <th className="text-left py-1 px-2">Window</th>
                                                <th className="text-left py-1 px-2">XGB</th>
                                                <th className="text-left py-1 px-2">LGB</th>
                                                <th className="text-left py-1 px-2">Ensemble</th>
                                                <th className="text-left py-1 px-2">Sharpe</th>
                                                <th className="text-left py-1 px-2">PF</th>
                                                <th className="text-left py-1 px-2">WR Long</th>
                                                <th className="text-left py-1 px-2">WR Short</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {Object.entries(m.v3AllWindows || {}).map(([window, w]: [string, any]) => (
                                                <tr key={window} className={`border-b border-border/20 ${window === m.v3BestWindow ? 'bg-green-500/10 font-medium' : ''}`}>
                                                    <td className="py-1 px-2 font-mono">
                                                        {window}
                                                        {window === m.v3BestWindow && <span className="ml-1 text-green-600">*</span>}
                                                    </td>
                                                    <td className="py-1 px-2">{((w.xgb_acc || 0) * 100).toFixed(1)}%</td>
                                                    <td className="py-1 px-2">{((w.lgb_acc || 0) * 100).toFixed(1)}%</td>
                                                    <td className="py-1 px-2">{((w.ens_acc || 0) * 100).toFixed(1)}%</td>
                                                    <td className="py-1 px-2">{(w.sharpe || 0).toFixed(2)}</td>
                                                    <td className="py-1 px-2">{(w.profit_factor || 0).toFixed(2)}</td>
                                                    <td className="py-1 px-2">{((w.win_rate_long || 0) * 100).toFixed(1)}%</td>
                                                    <td className="py-1 px-2">{((w.win_rate_short || 0) * 100).toFixed(1)}%</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            ))}
                        </div>
                    </details>
                )}

                <div className="mt-4 p-3 rounded-lg bg-purple-500/10 border border-purple-500/30">
                    <p className="text-xs text-purple-600">
                        <strong>V3 Pipeline:</strong> Regression-based prediction with 5 window configurations.
                        Best window selected by highest Sharpe ratio. WR Long/Short = win rate when predicting up/down.
                        Profit Factor = sum(wins) / sum(losses). Click "Show all window configs" for full breakdown.
                    </p>
                </div>
            </div>
        );
    }

    // Engine mode (original table)
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
                        <th className="text-left py-3 px-4 font-medium" title="Entry Quality Score">Quality</th>
                        <th className="text-left py-3 px-4 font-medium" title="Position Size (USD)">Pos. Size</th>
                        <th className="text-left py-3 px-4 font-medium" title="Risk %">Risk %</th>
                    </tr>
                </thead>
                <tbody>
                    {metrics.map((m, idx) => (
                        <tr key={m.symbol} className={`border-b border-border/30 hover:bg-secondary/20 ${idx % 2 === 0 ? 'bg-secondary/5' : ''}`}>
                            <td className="py-3 px-4 font-mono font-medium">{m.symbol}</td>

                            <td className="py-3 px-4">
                                <AccuracyCell value={m.modelPredictions?.xgb_accuracy} />
                            </td>

                            <td className="py-3 px-4">
                                <AccuracyCell value={m.modelPredictions?.lgb_accuracy} />
                            </td>

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

                            <td className="py-3 px-4">
                                {m.qualityScore != null ? (
                                    <span className={`font-medium ${m.qualityScore >= 0.70 ? 'text-green-500' : m.qualityScore >= 0.60 ? 'text-yellow-600' : 'text-red-500'}`}>
                                        {(m.qualityScore * 100).toFixed(0)}%
                                    </span>
                                ) : (
                                    <span className="text-muted-foreground text-xs">N/A</span>
                                )}
                            </td>

                            <td className="py-3 px-4">
                                {m.positionSizeUsd != null && m.positionSizeUsd > 0 ? (
                                    <span className="text-blue-600 font-medium">
                                        ${m.positionSizeUsd.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                    </span>
                                ) : (
                                    <span className="text-muted-foreground text-xs">—</span>
                                )}
                            </td>

                            <td className="py-3 px-4">
                                {m.riskPct != null && m.riskPct > 0 ? (
                                    <span className={`font-medium ${m.riskPct <= 0.01 ? 'text-green-500' : m.riskPct <= 0.02 ? 'text-yellow-600' : 'text-red-500'}`}>
                                        {(m.riskPct * 100).toFixed(2)}%
                                    </span>
                                ) : (
                                    <span className="text-muted-foreground text-xs">—</span>
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
                <p className="text-xs text-green-600 mt-2">
                    <strong>Phase 1 Metrics:</strong> <span className="font-mono">Quality</span> = Entry quality score ({'>'}70% to enter),
                    <span className="font-mono">Pos. Size</span> = Kelly Criterion position sizing,
                    <span className="font-mono">Risk %</span> = Risk per trade (target: {'≤'}1%).
                </p>
            </div>
        </div>
    );
};

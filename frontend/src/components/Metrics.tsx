/**
 * Metrics Component
 * Top-level metrics cards showing portfolio overview
 */

import React from 'react';
import { DollarSign, TrendingUp, Target, Brain } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MetricsProps {
    pipelineResult: any;
    wallet: any;
}

export const Metrics: React.FC<MetricsProps> = ({ pipelineResult, wallet }) => {
    // Calculate metrics
    const totalValue = wallet?.balance || 100000;
    const portfolioReturn = pipelineResult?.backtest_results
        ? Object.values(pipelineResult.backtest_results).reduce((acc: number, result: any) =>
            acc + (result.total_return || 0), 0) / Object.keys(pipelineResult.backtest_results).length
        : 0;

    const bestStock = pipelineResult?.backtest_results
        ? Object.entries(pipelineResult.backtest_results)
            .reduce((best: any, [symbol, data]: [string, any]) => {
                // Skip entries with errors or missing return data
                if (data.error || typeof data.total_return !== 'number') return best;

                return (data.total_return) > (best.return)
                    ? { symbol, return: data.total_return }
                    : best;
            }, { symbol: '-', return: -Infinity })
        : { symbol: '-', return: 0 };

    // Calculate ML accuracy only from valid results
    const validAccuracyResults = pipelineResult?.backtest_results
        ? Object.values(pipelineResult.backtest_results).filter((r: any) => r.model_predictions?.ensemble_accuracy !== undefined && r.model_predictions?.ensemble_accuracy !== null)
        : [];

    const mlAccuracy = validAccuracyResults.length > 0
        ? validAccuracyResults.reduce((acc: number, result: any) =>
            acc + (result.model_predictions?.ensemble_accuracy || 0), 0) / validAccuracyResults.length
        : 0;

    const metrics = [
        {
            label: 'Total Value',
            value: `₹${(totalValue / 1000).toFixed(1)}K`,
            subValue: wallet ? `${wallet.holdings_count || 0} holdings` : null,
            icon: DollarSign,
            isPositive: true,
        },
        {
            label: 'Portfolio Return',
            value: portfolioReturn ? `${(portfolioReturn * 100).toFixed(2)}%` : 'N/A',
            subValue: pipelineResult ? 'Last backtest' : null,
            icon: TrendingUp,
            isPositive: portfolioReturn >= 0,
            isNegative: portfolioReturn < 0,
        },
        {
            label: 'Best Stock',
            value: bestStock.symbol,
            subValue: bestStock.return !== 0 ? `${(bestStock.return * 100).toFixed(2)}%` : null,
            icon: Target,
            isPositive: bestStock.return > 0,
        },
        {
            label: 'ML Accuracy',
            value: mlAccuracy ? `${(mlAccuracy * 100).toFixed(1)}%` : 'N/A',
            subValue: 'Ensemble model',
            icon: Brain,
            isPositive: mlAccuracy >= 0.6,
            isNegative: mlAccuracy < 0.5,
        },
    ];

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {metrics.map((metric, idx) => {
                const Icon = metric.icon;
                return (
                    <div
                        key={idx}
                        className="bg-secondary/30 rounded-lg p-4 border border-border hover:border-primary/50 transition-all"
                    >
                        <div className="flex items-start justify-between mb-2">
                            <div className="p-2 rounded-lg bg-primary/10">
                                <Icon className="h-4 w-4 text-primary" />
                            </div>
                        </div>
                        <div className="text-xs text-muted-foreground mb-1">{metric.label}</div>
                        <div className={cn('text-2xl font-bold', {
                            'text-green-500': metric.isPositive,
                            'text-red-500': metric.isNegative,
                        })}>
                            {metric.value}
                        </div>
                        {metric.subValue && (
                            <div className="text-xs text-muted-foreground mt-1">{metric.subValue}</div>
                        )}
                    </div>
                );
            })}
        </div>
    );
};

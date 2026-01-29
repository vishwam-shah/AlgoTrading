/**
 * Signals Component
 * Trading signals table with execute buttons
 */

import React from 'react';
import { ArrowUpRight, ArrowDownRight, MinusCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Signal {
    action: string;
    confidence: number;
    direction_probability: number;
    expected_return: number;
    current_price: number;
}

interface SignalsProps {
    signals: Record<string, Signal>;
    onExecuteTrade: (symbol: string) => void;
    isTrading: boolean;
}

export const Signals: React.FC<SignalsProps> = ({ signals, onExecuteTrade, isTrading }) => {
    if (!signals || Object.keys(signals).length === 0) {
        return (
            <div className="text-center py-8 text-muted-foreground">
                <p className="text-sm">Run a pipeline to generate trading signals</p>
            </div>
        );
    }

    const signalEntries = Object.entries(signals);

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-border">
                        <th className="text-left py-3 px-4 font-medium">Symbol</th>
                        <th className="text-left py-3 px-4 font-medium">Signal</th>
                        <th className="text-left py-3 px-4 font-medium">Confidence</th>
                        <th className="text-left py-3 px-4 font-medium">Expected Return</th>
                        <th className="text-left py-3 px-4 font-medium">Price</th>
                        <th className="text-right py-3 px-4 font-medium">Action</th>
                    </tr>
                </thead>
                <tbody>
                    {signalEntries.map(([symbol, signal], idx) => (
                        <tr
                            key={symbol}
                            className={cn(
                                'border-b border-border/30 hover:bg-secondary/20',
                                idx % 2 === 0 ? 'bg-secondary/5' : ''
                            )}
                        >
                            <td className="py-3 px-4 font-mono font-medium">{symbol}</td>
                            <td className="py-3 px-4">
                                <span
                                    className={cn(
                                        'px-2 py-0.5 rounded text-xs font-medium inline-flex items-center gap-1',
                                        signal.action === 'BUY'
                                            ? 'bg-green-500/20 text-green-600'
                                            : signal.action === 'SELL'
                                                ? 'bg-red-500/20 text-red-600'
                                                : 'bg-yellow-500/20 text-yellow-600'
                                    )}
                                >
                                    {signal.action === 'BUY' && <ArrowUpRight className="h-3 w-3" />}
                                    {signal.action === 'SELL' && <ArrowDownRight className="h-3 w-3" />}
                                    {signal.action === 'HOLD' && <MinusCircle className="h-3 w-3" />}
                                    {signal.action}
                                </span>
                            </td>
                            <td className="py-3 px-4">
                                <span
                                    className={cn(
                                        'font-medium',
                                        signal.confidence >= 0.65
                                            ? 'text-green-500'
                                            : signal.confidence <= 0.55
                                                ? 'text-yellow-600'
                                                : ''
                                    )}
                                >
                                    {(signal.confidence * 100).toFixed(1)}%
                                </span>
                            </td>
                            <td className="py-3 px-4">
                                <span
                                    className={signal.expected_return >= 0 ? 'text-green-500' : 'text-red-500'}
                                >
                                    {(signal.expected_return * 100).toFixed(2)}%
                                </span>
                            </td>
                            <td className="py-3 px-4">₹{signal.current_price.toFixed(2)}</td>
                            <td className="py-3 px-4 text-right">
                                {signal.action !== 'HOLD' && (
                                    <button
                                        onClick={() => onExecuteTrade(symbol)}
                                        disabled={isTrading}
                                        className={cn(
                                            'px-3 py-1 rounded text-xs font-medium transition-colors',
                                            signal.action === 'BUY'
                                                ? 'bg-green-500 text-white hover:bg-green-600'
                                                : 'bg-red-500 text-white hover:bg-red-600',
                                            isTrading && 'opacity-50 cursor-not-allowed'
                                        )}
                                    >
                                        {isTrading ? (
                                            <Loader2 className="h-3 w-3 animate-spin" />
                                        ) : (
                                            'Execute'
                                        )}
                                    </button>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

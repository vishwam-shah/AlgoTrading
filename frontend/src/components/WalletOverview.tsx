/**
 * Wallet Overview Component
 * Detailed view of portfolio holdings and account balance
 */

import React from 'react';
import { Wallet, TrendingUp, TrendingDown, DollarSign, PieChart } from 'lucide-react';
import { cn } from '@/lib/utils';

interface PortfolioPosition {
    symbol: string;
    shares: number;
    avg_price: number;
    current_price: number;
    current_value: number;
    pnl: number;
    pnl_pct: number;
}

interface WalletData {
    balance: number;
    portfolio_value: number;
    total_value: number;
    total_pnl: number;
    total_pnl_pct: number;
    holdings_count: number;
    portfolio: Record<string, PortfolioPosition>;
}

interface WalletOverviewProps {
    wallet: WalletData | null;
}

export const WalletOverview: React.FC<WalletOverviewProps> = ({ wallet }) => {
    if (!wallet) {
        return (
            <div className="p-6 bg-secondary/20 rounded-lg border border-border text-center">
                <Wallet className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                <p className="text-muted-foreground">Please log in to view wallet details</p>
            </div>
        );
    }

    const { portfolio } = wallet;
    const holdings = Object.entries(portfolio || {});

    return (
        <div className="bg-card border border-border rounded-lg overflow-hidden">
            {/* Header */}
            <div className="p-6 border-b border-border flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Wallet className="h-5 w-5 text-primary" />
                    <h2 className="text-lg font-semibold">Portfolio Holdings</h2>
                </div>
                <div className="flex gap-4 text-sm">
                    <div>
                        <span className="text-muted-foreground block text-xs">Cash Balance</span>
                        <span className="font-mono font-medium">₹{wallet.balance.toFixed(2)}</span>
                    </div>
                    <div>
                        <span className="text-muted-foreground block text-xs">Invested</span>
                        <span className="font-mono font-medium">₹{wallet.portfolio_value.toFixed(2)}</span>
                    </div>
                    <div>
                        <span className="text-muted-foreground block text-xs">Total P&L</span>
                        <span className={cn(
                            "font-mono font-medium flex items-center gap-1",
                            wallet.total_pnl >= 0 ? "text-green-500" : "text-red-500"
                        )}>
                            {wallet.total_pnl >= 0 ? '+' : ''}₹{wallet.total_pnl.toFixed(2)}
                            <span className="text-xs">({wallet.total_pnl_pct.toFixed(2)}%)</span>
                        </span>
                    </div>
                </div>
            </div>

            {/* Holdings Table */}
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead className="bg-secondary/30 text-muted-foreground border-b border-border">
                        <tr>
                            <th className="text-left py-3 px-4 font-medium">Symbol</th>
                            <th className="text-right py-3 px-4 font-medium">Shares</th>
                            <th className="text-right py-3 px-4 font-medium">Avg Price</th>
                            <th className="text-right py-3 px-4 font-medium">Current</th>
                            <th className="text-right py-3 px-4 font-medium">Value</th>
                            <th className="text-right py-3 px-4 font-medium">P&L</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                        {holdings.length === 0 ? (
                            <tr>
                                <td colSpan={6} className="py-8 text-center text-muted-foreground">
                                    <PieChart className="h-8 w-8 mx-auto mb-2 opacity-50" />
                                    No active holdings
                                </td>
                            </tr>
                        ) : (
                            holdings.map(([symbol, position]) => (
                                <tr key={symbol} className="hover:bg-secondary/10 transition-colors">
                                    <td className="py-3 px-4 font-medium">{symbol}</td>
                                    <td className="py-3 px-4 text-right font-mono">{position.shares}</td>
                                    <td className="py-3 px-4 text-right font-mono text-muted-foreground">
                                        ₹{position.avg_price.toFixed(2)}
                                    </td>
                                    <td className="py-3 px-4 text-right font-mono">
                                        ₹{position.current_price.toFixed(2)}
                                    </td>
                                    <td className="py-3 px-4 text-right font-mono font-medium">
                                        ₹{position.current_value.toFixed(2)}
                                    </td>
                                    <td className="py-3 px-4 text-right">
                                        <div className={cn(
                                            "inline-flex items-center gap-1 font-mono",
                                            position.pnl >= 0 ? "text-green-500" : "text-red-500"
                                        )}>
                                            {position.pnl >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                                            {position.pnl_pct.toFixed(2)}%
                                        </div>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

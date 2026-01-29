/**
 * Header Component
 * Dashboard header with logo, title, and action buttons
 */

import React from 'react';
import { Brain, Settings, Sun, Moon } from 'lucide-react';
import { CSVExporter } from './CSVExporter';

interface HeaderProps {
    theme: string | undefined;
    onThemeToggle: () => void;
    onSettingsClick: () => void;
    pipelineResult: any;
    pipelineJobId: string | null;
    wallet?: any;
}

export const Header: React.FC<HeaderProps> = ({
    theme,
    onThemeToggle,
    onSettingsClick,
    pipelineResult,
    pipelineJobId,
    wallet
}) => {
    const [mounted, setMounted] = React.useState(false);

    React.useEffect(() => {
        setMounted(true);
    }, []);

    return (
        <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur">
            <div className="container mx-auto px-4 py-3">
                <div className="flex items-center justify-between">
                    {/* Logo & Title */}
                    <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
                            <Brain className="h-5 w-5 text-white" />
                        </div>
                        <div>
                            <h1 className="text-lg font-bold">Long-Term Equity Portfolio</h1>
                            <p className="text-xs text-muted-foreground">8-Step Factor-Based Investing Pipeline</p>
                        </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex items-center gap-4">
                        {/* Wallet Stats */}
                        {wallet && (
                            <div className="hidden md:flex items-center gap-4 mr-2 bg-secondary/30 px-3 py-1.5 rounded-lg border border-border/50">
                                <div className="text-right">
                                    <p className="text-[10px] text-muted-foreground uppercase font-semibold">Portfolio Value</p>
                                    <p className="font-mono font-medium text-sm">₹{wallet.portfolio_value?.toLocaleString('en-IN')}</p>
                                </div>
                                <div className="h-6 w-px bg-border/50" />
                                <div className="text-right">
                                    <p className="text-[10px] text-muted-foreground uppercase font-semibold">Balance</p>
                                    <p className="font-mono font-medium text-sm">₹{wallet.balance?.toLocaleString('en-IN')}</p>
                                </div>
                                {wallet.total_pnl !== undefined && (
                                    <>
                                        <div className="h-6 w-px bg-border/50" />
                                        <div className="text-right">
                                            <p className="text-[10px] text-muted-foreground uppercase font-semibold">PnL</p>
                                            <p className={`font-mono font-medium text-sm ${wallet.total_pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                                {wallet.total_pnl >= 0 ? '+' : ''}₹{Math.abs(wallet.total_pnl).toLocaleString('en-IN')}
                                            </p>
                                        </div>
                                    </>
                                )}
                            </div>
                        )}

                        {/* CSV Export */}
                        {pipelineResult && (
                            <CSVExporter
                                pipelineId={pipelineJobId || 'latest'}
                                backtestResults={pipelineResult.backtest_results || {}}
                                signals={pipelineResult.signals || {}}
                            />
                        )}

                        {/* Settings Button */}
                        <button
                            onClick={onSettingsClick}
                            className="p-2 rounded-lg border border-border hover:bg-secondary/50 transition-colors"
                            title="Pipeline Settings"
                        >
                            <Settings className="h-5 w-5" />
                        </button>

                        {/* Theme Toggle */}
                        <button
                            onClick={onThemeToggle}
                            className="p-2 rounded-lg border border-border hover:bg-secondary/50 transition-colors"
                            title="Toggle Theme"
                        >
                            {mounted && (theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />)}
                        </button>
                    </div>
                </div>
            </div>
        </header>
    );
};

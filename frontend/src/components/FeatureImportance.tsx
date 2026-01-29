/**
 * Feature Importance Component
 * Displays the most important features driving the ML model's predictions
 */

import React from 'react';
import { BarChart3, Info } from 'lucide-react';
import { cn } from '@/lib/utils';

interface FeatureImportanceProps {
    featureImportance: Array<{
        feature: string;
        importance: number;
    }>;
}

export const FeatureImportance: React.FC<FeatureImportanceProps> = ({ featureImportance }) => {
    if (!featureImportance || featureImportance.length === 0) {
        return null;
    }

    // Sort by importance descending and take top 20
    const sortedFeatures = [...featureImportance]
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 20);

    const maxImportance = Math.max(...sortedFeatures.map(f => f.importance));

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="h-5 w-5 text-primary" />
                <h3 className="font-semibold text-lg">Model Feature Importance</h3>
                <div className="group relative ml-auto">
                    <Info className="h-4 w-4 text-muted-foreground cursor-help" />
                    <div className="absolute right-0 w-64 p-2 bg-secondary border border-border rounded-lg text-xs text-muted-foreground shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        Shows which market factors are most influential in the model's decision making process.
                    </div>
                </div>
            </div>

            <div className="space-y-3">
                {sortedFeatures.map((item, idx) => (
                    <div key={idx} className="space-y-1">
                        <div className="flex justify-between text-sm">
                            <span className="font-mono text-muted-foreground">{item.feature}</span>
                            <span className="font-medium">{(item.importance * 100).toFixed(2)}%</span>
                        </div>
                        <div className="h-2 w-full bg-secondary/50 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-primary/80 rounded-full transition-all duration-500"
                                style={{ width: `${(item.importance / maxImportance) * 100}%` }}
                            />
                        </div>
                    </div>
                ))}
            </div>

            <div className="pt-4 border-t border-border mt-4">
                <p className="text-xs text-muted-foreground text-center">
                    Top {sortedFeatures.length} predictive features out of {featureImportance.length} total
                </p>
            </div>
        </div>
    );
};

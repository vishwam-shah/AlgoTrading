/**
 * Chart Component
 * Equity curve visualization
 */

import React from 'react';
import {
    ResponsiveContainer,
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartsTooltip,
} from 'recharts';

interface ChartProps {
    equityCurve: { date: string; equity: number }[];
    title?: string;
}

export const Chart: React.FC<ChartProps> = ({ equityCurve, title = 'Equity Curve' }) => {
    if (!equityCurve || equityCurve.length === 0) {
        return (
            <div className="h-64 flex items-center justify-center text-muted-foreground">
                <p className="text-sm">No data available</p>
            </div>
        );
    }

    return (
        <div>
            {title && <h3 className="text-sm font-medium mb-4">{title}</h3>}
            <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={equityCurve}>
                    <defs>
                        <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="rgb(59, 130, 246)" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="rgb(59, 130, 246)" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgb(100, 100, 100)" opacity={0.1} />
                    <XAxis
                        dataKey="date"
                        stroke="rgb(150, 150, 150)"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => {
                            const date = new Date(value);
                            return `${date.getMonth() + 1}/${date.getDate()}`;
                        }}
                    />
                    <YAxis
                        stroke="rgb(150, 150, 150)"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => `₹${(value / 1000).toFixed(0)}K`}
                    />
                    <RechartsTooltip
                        contentStyle={{
                            backgroundColor: 'rgb(30, 30, 30)',
                            border: '1px solid rgb(60, 60, 60)',
                            borderRadius: '8px',
                        }}
                        labelStyle={{ color: 'rgb(200, 200, 200)' }}
                        formatter={(value: any) => [`₹${value.toFixed(2)}`, 'Equity']}
                    />
                    <Area
                        type="monotone"
                        dataKey="equity"
                        stroke="rgb(59, 130, 246)"
                        strokeWidth={2}
                        fill="url(#equityGradient)"
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
};

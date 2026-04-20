'use client';
import { useEffect, useRef } from 'react';
import { createChart, ColorType, LineStyle, AreaSeries, LineSeries, type IChartApi } from 'lightweight-charts';

interface EquityPoint { date: string; value: number; }

interface EquityCurveChartProps {
  data: EquityPoint[];
  benchmark?: EquityPoint[];
  title?: string;
  height?: number;
  capital?: number;
}

export default function EquityCurveChart({
  data,
  benchmark,
  title = 'Portfolio Equity Curve',
  height = 260,
  capital = 500000,
}: EquityCurveChartProps) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!ref.current || !data.length) return;

    const isDark = document.documentElement.classList.contains('dark');
    const bg   = isDark ? '#0a0a0f' : '#ffffff';
    const grid = isDark ? '#1a1a2e' : '#f0f0f0';
    const text = isDark ? '#9ca3af' : '#6b7280';

    const chart = createChart(ref.current, {
      layout:  { background: { type: ColorType.Solid, color: bg }, textColor: text },
      grid:    { vertLines: { color: grid }, horzLines: { color: grid } },
      rightPriceScale: { borderColor: grid },
      timeScale: { borderColor: grid },
      width:  ref.current.clientWidth,
      height,
    });
    chartRef.current = chart;

    // Strategy equity line (v5 API)
    const strategy = chart.addSeries(AreaSeries, {
      lineColor:   '#6366f1',
      topColor:    '#6366f140',
      bottomColor: '#6366f100',
      lineWidth: 2,
      priceFormat: { type: 'price', precision: 0, minMove: 1 },
    });
    strategy.setData(data.map(d => ({ time: d.date as any, value: d.value })));

    // Baseline (capital)
    const baseline = chart.addSeries(LineSeries, {
      color:     '#6b7280',
      lineStyle: LineStyle.Dashed,
      lineWidth: 1,
      priceFormat: { type: 'price', precision: 0, minMove: 1 },
    });
    baseline.setData(data.map(d => ({ time: d.date as any, value: capital })));

    // Nifty50 benchmark if provided
    if (benchmark?.length) {
      const bench = chart.addSeries(LineSeries, {
        color:     '#f59e0b',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
      });
      bench.setData(benchmark.map(d => ({ time: d.date as any, value: d.value })));
    }

    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      if (ref.current) chart.applyOptions({ width: ref.current.clientWidth });
    });
    ro.observe(ref.current);

    return () => { ro.disconnect(); chart.remove(); };
  }, [data, benchmark, height, capital]);

  if (!data.length) return (
    <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
      No equity data yet
    </div>
  );

  const lastVal  = data[data.length - 1]?.value ?? capital;
  const pnl      = lastVal - capital;
  const pnlPct   = (pnl / capital) * 100;

  return (
    <div className="w-full rounded-xl overflow-hidden border border-border bg-card">
      <div className="px-4 py-2 border-b border-border flex items-center justify-between">
        <span className="text-sm font-medium">{title}</span>
        <span className={`text-sm font-mono font-bold ${pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
          {pnl >= 0 ? '+' : ''}₹{pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
          {' '}({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%)
        </span>
      </div>
      <div ref={ref} style={{ height }} />
    </div>
  );
}

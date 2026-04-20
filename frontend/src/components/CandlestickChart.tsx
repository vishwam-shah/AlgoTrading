'use client';
import { useEffect, useRef } from 'react';
import {
  createChart,
  createSeriesMarkers,
  ColorType,
  CrosshairMode,
  CandlestickSeries,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type HistogramData,
} from 'lightweight-charts';

interface OHLCVBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface SignalMarker {
  date: string;
  direction: 'UP' | 'DOWN';
  prob_up?: number;
}

interface CandlestickChartProps {
  data: OHLCVBar[];
  signals?: SignalMarker[];
  symbol: string;
  height?: number;
}

export default function CandlestickChart({
  data,
  signals = [],
  symbol,
  height = 420,
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<IChartApi | null>(null);
  const candleRef    = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeRef    = useRef<ISeriesApi<'Histogram'> | null>(null);

  useEffect(() => {
    if (!containerRef.current || !data.length) return;

    const isDark = document.documentElement.classList.contains('dark');
    const bg     = isDark ? '#0a0a0f' : '#ffffff';
    const grid   = isDark ? '#1a1a2e' : '#f0f0f0';
    const text   = isDark ? '#9ca3af' : '#6b7280';
    const border = isDark ? '#1e1e2e' : '#e5e7eb';

    const chart = createChart(containerRef.current, {
      layout:  { background: { type: ColorType.Solid, color: bg }, textColor: text },
      grid:    { vertLines: { color: grid }, horzLines: { color: grid } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: border },
      timeScale: { borderColor: border, timeVisible: true },
      width:  containerRef.current.clientWidth,
      height: height - 80,
    });
    chartRef.current = chart;

    // Candlestick series (v5 API)
    const candle = chart.addSeries(CandlestickSeries, {
      upColor:        '#22c55e',
      downColor:      '#ef4444',
      borderUpColor:  '#22c55e',
      borderDownColor:'#ef4444',
      wickUpColor:    '#22c55e',
      wickDownColor:  '#ef4444',
    });
    candleRef.current = candle;

    const candleData: CandlestickData[] = data.map(d => ({
      time: d.date as any,
      open: d.open, high: d.high, low: d.low, close: d.close,
    }));
    candle.setData(candleData);

    // Signal markers
    if (signals.length) {
      const markers = signals
        .filter(s => data.find(d => d.date === s.date))
        .map(s => ({
          time:     s.date as any,
          position: s.direction === 'UP' ? 'belowBar' : 'aboveBar',
          color:    s.direction === 'UP' ? '#22c55e' : '#ef4444',
          shape:    s.direction === 'UP' ? 'arrowUp'  : 'arrowDown',
          text:     s.prob_up ? `${(s.prob_up * 100).toFixed(0)}%` : '',
          size: 1,
        } as any));
      createSeriesMarkers(candle, markers);
    }

    // Volume pane (v5 API)
    const volume = chart.addSeries(HistogramSeries, {
      color: '#6366f1',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      scaleMargins: { top: 0.85, bottom: 0 },
    } as any);
    volumeRef.current = volume;

    const volData: HistogramData[] = data.map(d => ({
      time:  d.date as any,
      value: d.volume,
      color: d.close >= d.open ? '#22c55e55' : '#ef444455',
    }));
    volume.setData(volData);

    chart.timeScale().fitContent();

    // Resize observer
    const ro = new ResizeObserver(() => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth });
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [data, signals, height]);

  return (
    <div className="w-full rounded-xl overflow-hidden border border-border bg-card">
      <div className="px-4 py-2 border-b border-border flex items-center gap-2">
        <span className="font-mono font-bold text-sm">{symbol}</span>
        <span className="text-xs text-muted-foreground">Daily OHLCV · V3 Pipeline Data</span>
        {signals.length > 0 && (
          <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            {signals.filter(s => s.direction === 'UP').length} ▲ signals
          </span>
        )}
      </div>
      <div ref={containerRef} style={{ height: height - 80 }} />
    </div>
  );
}

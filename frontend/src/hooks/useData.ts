'use client';
import { useState, useEffect, useCallback, useRef } from 'react';

export async function apiFetch<T>(path: string, timeoutMs = 8000): Promise<T | null> {
  const ctrl = new AbortController();
  const tid  = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(path, { cache: 'no-store', signal: ctrl.signal });
    clearTimeout(tid);
    return r.ok ? r.json() : null;
  } catch { clearTimeout(tid); return null; }
}

export function usePolling<T>(path: string, intervalMs = 30000) {
  const [data, setData]       = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(false);

  const fetch_ = useCallback(async () => {
    const d = await apiFetch<T>(path);
    if (d) { setData(d); setError(false); }
    else setError(true);
    setLoading(false);
  }, [path]);

  useEffect(() => {
    fetch_();
    const t = setInterval(fetch_, intervalMs);
    return () => clearInterval(t);
  }, [fetch_, intervalMs]);

  return { data, loading, error, refresh: fetch_ };
}

export function useLivePrices(symbols: string[]) {
  const [prices, setPrices]       = useState<Record<string, number>>({});
  const [prev, setPrev]           = useState<Record<string, number>>({});
  const [connected, setConnected] = useState(false);
  const wsRef      = useRef<WebSocket | null>(null);
  const pricesRef  = useRef<Record<string, number>>({});
  const symbolsRef = useRef<string[]>(symbols);

  // When symbols change, update ref and push new list to open connection (no reconnect)
  useEffect(() => {
    symbolsRef.current = symbols;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN && symbols.length) {
      ws.send(JSON.stringify({ symbols }));
    }
  }, [symbols]);

  // Connect once on mount; reconnect on close, never on symbol changes
  useEffect(() => {
    let cancelled = false;

    function connect() {
      if (cancelled) return;
      const ws = new WebSocket('ws://localhost:8000/ws/live-prices');
      wsRef.current = ws;
      ws.onopen = () => {
        if (cancelled) { ws.close(); return; }
        setConnected(true);
        if (symbolsRef.current.length) ws.send(JSON.stringify({ symbols: symbolsRef.current }));
      };
      ws.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data);
          if (d.prices) {
            setPrev(pricesRef.current);
            pricesRef.current = d.prices;
            setPrices(d.prices);
          }
        } catch {}
      };
      ws.onclose = () => { setConnected(false); if (!cancelled) setTimeout(connect, 5000); };
      ws.onerror = () => ws.close();
    }

    connect();
    return () => { cancelled = true; wsRef.current?.close(); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const change = useCallback((sym: string) => {
    const p = prices[sym], q = prev[sym];
    if (!p || !q) return null;
    return ((p - q) / q) * 100;
  }, [prices, prev]);

  return { prices, change, connected };
}

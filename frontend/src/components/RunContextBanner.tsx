'use client';
import { History } from 'lucide-react';

interface Props { runId: string; onClear: () => void; }

export default function RunContextBanner({ runId, onClear }: Props) {
  return (
    <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-xs text-indigo-300">
      <History className="h-3.5 w-3.5 flex-shrink-0" />
      Viewing historical run <span className="font-mono font-bold">{runId}</span>
      <button onClick={onClear} className="ml-auto text-indigo-400 hover:text-indigo-200 underline">
        Switch to latest →
      </button>
    </div>
  );
}

import { cn } from '@/lib/utils';
import { type LucideIcon } from 'lucide-react';

interface StatCardProps {
  label: string;
  value: string | number;
  sub?: string;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  highlight?: boolean;
  className?: string;
}

export default function StatCard({ label, value, sub, icon: Icon, trend, highlight, className }: StatCardProps) {
  return (
    <div className={cn(
      'rounded-xl border border-border bg-card p-4 flex flex-col gap-1',
      highlight && 'border-primary/30 bg-primary/5',
      className,
    )}>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground uppercase tracking-wide font-medium">{label}</span>
        {Icon && <Icon className="h-4 w-4 text-muted-foreground/50" />}
      </div>
      <div className={cn(
        'text-2xl font-bold font-mono tabular-nums',
        trend === 'up'   ? 'text-green-400'
        : trend === 'down' ? 'text-red-400'
        : 'text-foreground'
      )}>
        {value}
      </div>
      {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}

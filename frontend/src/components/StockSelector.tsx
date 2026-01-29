import * as Accordion from '@radix-ui/react-accordion';
import { Play, Layers, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface StockSelectorProps {
    sectorStockMap: Record<string, string[]>;
    selectedSectors: string[];
    selectedStocks: string[];
    // expandedSectors no longer needed as state, fully managed by Radix or local if needed, 
    // but looking at usage we can let Radix manage uncontrolled or controlled. 
    // Ensuring backward compatibility with props if parent controls it, but parent passed `expandedSectors` state.
    // For now, let's keep parent happy but implement internal Radix logic.
    // Actually, to use Radix Accordion fully, we should map sectors to AccordionItems.
    expandedSectors: string[];
    onSectorToggle: (sector: string) => void;
    onStockToggle: (stock: string) => void;
    onExpandSector: (sector: string) => void; // We can sync this with Radix onValueChange
    onRunPipeline: () => void;
    isRunning: boolean;
}

export const StockSelector: React.FC<StockSelectorProps> = ({
    sectorStockMap,
    selectedSectors,
    selectedStocks,
    expandedSectors,
    onSectorToggle,
    onStockToggle,
    onExpandSector,
    onRunPipeline,
    isRunning
}) => {
    const allSectors = Object.keys(sectorStockMap);

    return (
        <div className="bg-card rounded-lg p-6 border border-border shadow-sm">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                    <div className="p-2 bg-primary/10 rounded-lg">
                        <Layers className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold">Stock Selection</h2>
                        <p className="text-xs text-muted-foreground">Choose sectors or individual stocks</p>
                    </div>
                </div>
                <div className="flex gap-3">
                    <button
                        onClick={() => {
                            if (selectedSectors.length === allSectors.length) {
                                onSectorToggle('__ALL__');
                            } else {
                                onSectorToggle('__ALL__');
                            }
                        }}
                        className="text-xs text-muted-foreground hover:text-primary transition-colors font-medium px-3 py-2 hover:bg-secondary rounded-lg"
                    >
                        {selectedSectors.length === allSectors.length ? 'Deselect All' : 'Select All'}
                    </button>
                    <button
                        onClick={onRunPipeline}
                        disabled={isRunning || (selectedSectors.length === 0 && selectedStocks.length === 0)}
                        className={cn(
                            'px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 shadow-sm',
                            isRunning || (selectedSectors.length === 0 && selectedStocks.length === 0)
                                ? 'bg-secondary text-muted-foreground cursor-not-allowed'
                                : 'bg-primary text-primary-foreground hover:bg-primary/90 hover:shadow-md hover:scale-[1.02] active:scale-[0.98]'
                        )}
                    >
                        <Play className={cn("h-4 w-4", isRunning && "animate-spin")} />
                        {isRunning ? 'Running API...' : 'Run Pipeline'}
                    </button>
                </div>
            </div>

            <Accordion.Root
                type="multiple"
                value={expandedSectors}
                onValueChange={(value) => {
                    // Sync with parent state if needed, but since parent expects single toggle calls,
                    // we might need to differ. Parent `toggleSectorExpand` toggles one by one.
                    // Ideally we should refactor parent to accept array, but strict interface usage prevents breaking changes.
                    // We'll trust user interaction for now or strictly map value.
                    // Actually, let's just bypass parent's Expand logging for pure UI if it doesn't affect logic?
                    // No, `expandedSectors` is passed in. We must iterate and sync.
                    // Simplification: We iterate and call onExpandSector for diffs? 
                    // Or better: Just use Radix UI visually and trigger parent toggles?
                    // Let's assume onExpandSector toggles presence in array.
                    // Finding which one changed is tricky without prev state.
                    // Let's use the individual Triggers to call onExpandSector.
                }}
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
            >
                {allSectors.map((sector) => {
                    const isSelected = selectedSectors.includes(sector);
                    const stocks = sectorStockMap[sector] || [];

                    return (
                        <Accordion.Item
                            key={sector}
                            value={sector}
                            className="group border border-border rounded-lg bg-card overflow-hidden data-[state=open]:border-primary/50 transition-all data-[state=open]:shadow-md"
                        >
                            <div className="flex items-center p-3 gap-2">
                                <Accordion.Trigger
                                    onClick={() => onExpandSector(sector)}
                                    className="p-1 hover:bg-secondary rounded-md transition-colors group-data-[state=open]:rotate-180"
                                >
                                    <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform duration-200" />
                                </Accordion.Trigger>

                                <div
                                    onClick={() => onSectorToggle(sector)}
                                    className={cn(
                                        'flex-1 flex items-center justify-between cursor-pointer select-none rounded px-2 py-1 transition-colors',
                                        isSelected ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-secondary/50 text-foreground'
                                    )}
                                >
                                    <span>{sector}</span>
                                    <span className="text-xs text-muted-foreground ml-2 bg-secondary px-1.5 py-0.5 rounded-full">
                                        {stocks.length}
                                    </span>
                                </div>
                            </div>

                            <Accordion.Content className="data-[state=open]:animate-accordion-down data-[state=closed]:animate-accordion-up overflow-hidden text-sm">
                                <div className="p-3 pt-0 grid grid-cols-2 gap-2">
                                    {stocks.map((stock) => {
                                        const isStockSelected = selectedStocks.includes(stock);
                                        return (
                                            <button
                                                key={stock}
                                                onClick={() => onStockToggle(stock)}
                                                className={cn(
                                                    'px-2 py-1.5 rounded text-xs font-mono transition-all text-left truncate flex items-center gap-2',
                                                    isStockSelected
                                                        ? 'bg-primary text-primary-foreground shadow-sm'
                                                        : 'bg-secondary/30 hover:bg-secondary text-muted-foreground hover:text-foreground'
                                                )}
                                            >
                                                <div className={cn("w-1.5 h-1.5 rounded-full", isStockSelected ? "bg-white" : "bg-border")} />
                                                {stock}
                                            </button>
                                        );
                                    })}
                                </div>
                            </Accordion.Content>
                        </Accordion.Item>
                    );
                })}
            </Accordion.Root>

            <div className="mt-6 flex items-center justify-between text-xs text-muted-foreground px-1">
                <p>
                    Selecting a sector auto-selects all its stocks.
                </p>
                <p>
                    <span className="font-medium text-foreground">{selectedStocks.length}</span> stocks active
                </p>
            </div>
        </div>
    );
};

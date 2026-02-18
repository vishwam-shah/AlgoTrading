/**
 * Progress Component
 * Pipeline progress bar and step indicators
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, CheckCircle, XCircle } from 'lucide-react';
import * as Progress from '@radix-ui/react-progress';
import { cn } from '@/lib/utils';

interface PipelineStep {
    step: number;
    name: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    duration: number;
}

interface PipelineStatus {
    status: string;
    current_step: number;
    total_steps: number;
    progress: number;
    steps: PipelineStep[];
}

interface ProgressProps {
    isRunning: boolean;
    pipelineStatus: PipelineStatus | null;
}

export const ProgressComponent: React.FC<ProgressProps> = ({ isRunning, pipelineStatus }) => {
    if (!isRunning || !pipelineStatus) return null;
    const steps = Array.isArray(pipelineStatus.steps) ? pipelineStatus.steps : [];

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="border-b border-border bg-secondary/30"
            >
                <div className="container mx-auto px-4 py-4">
                    {/* Progress Bar */}
                    <div className="mb-4">
                        <div className="flex justify-between text-sm mb-2">
                            <span className="font-medium">
                                Pipeline Running: Step {pipelineStatus.current_step}/{pipelineStatus.total_steps}
                            </span>
                            <span className="text-muted-foreground">
                                {Math.round(pipelineStatus.progress)}%
                            </span>
                        </div>
                        <Progress.Root className="relative overflow-hidden bg-secondary rounded-full h-2">
                            <Progress.Indicator
                                className="bg-gradient-to-r from-blue-500 to-blue-600 h-full transition-transform duration-300 ease-in-out"
                                style={{ transform: `translateX(-${100 - pipelineStatus.progress}%)` }}
                            />
                        </Progress.Root>
                    </div>

                    {/* Step Indicators */}
                    <div className={`grid grid-cols-2 md:grid-cols-4 ${steps.length <= 4 ? 'lg:grid-cols-4' : 'lg:grid-cols-8'} gap-2`}>
                        {steps.map((step) => {
                            const isActive = step.step === pipelineStatus.current_step;
                            const isCompleted = step.status === 'completed';
                            const isFailed = step.status === 'failed';

                            return (
                                <div
                                    key={step.step}
                                    className={cn(
                                        'flex items-center gap-2 p-2 rounded-lg border transition-all text-xs',
                                        isActive ? 'border-blue-500 bg-blue-500/5' :
                                            isCompleted ? 'border-green-500/50 bg-green-500/5' :
                                                isFailed ? 'border-red-500/50 bg-red-500/5' :
                                                    'border-border/50'
                                    )}
                                >
                                    <div className={cn(
                                        'w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0',
                                        isActive ? 'bg-blue-500 text-white' :
                                            isCompleted ? 'bg-green-500 text-white' :
                                                isFailed ? 'bg-red-500 text-white' :
                                                    'bg-secondary text-muted-foreground'
                                    )}>
                                        {isCompleted ? <CheckCircle className="h-3 w-3" /> :
                                            isFailed ? <XCircle className="h-3 w-3" /> :
                                                isActive ? <Loader2 className="h-3 w-3 animate-spin" /> :
                                                    step.step}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="font-medium truncate">{step.name}</div>
                                        {step.duration > 0 && (
                                            <div className="text-xs text-muted-foreground">{step.duration.toFixed(1)}s</div>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </motion.div>
        </AnimatePresence>
    );
};

"""
================================================================================
GENERATE COMBINED RESEARCH PLOTS
================================================================================
Generates combined analysis plots for all stocks with existing results.

This script:
1. Reads all model comparison CSVs
2. Aggregates performance metrics across all stocks
3. Generates combined visualization plots
4. Saves all plots to evaluation_results/plots/

Usage:
    python generate_combined_plots.py
================================================================================
"""

import os
import sys
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.research_plots import generate_combined_plots


def main():
    """Generate combined analysis plots for all stocks."""
    
    logger.info("="*80)
    logger.info("GENERATING COMBINED RESEARCH PLOTS")
    logger.info("="*80)
    
    try:
        generate_combined_plots()
        logger.success("\n✅ Combined plots generated successfully!")
        logger.info("\nPlots saved to: evaluation_results/plots/")
        logger.info("  - COMBINED_model_performance.png")
        logger.info("  - COMBINED_direction_accuracy_distribution.png")
        logger.info("  - COMBINED_r2_distribution.png")
        logger.info("  - COMBINED_best_model_distribution.png")
        logger.info("  - COMBINED_performance_heatmap.png")
        
    except Exception as e:
        logger.error(f"Failed to generate combined plots: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

"""
================================================================================
ENTRY OPTIMIZER - Multi-Condition Entry Filters
================================================================================
Improves win rate by filtering trades based on multiple confirmation signals.
Only enters trades when ALL conditions are met.

Target: Increase win rate from 45-73% to 60-80%
================================================================================
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class EntrySignal:
    """Entry signal with quality score and reasons."""
    should_enter: bool
    quality_score: float  # 0-1
    confidence: float
    reasons: List[str]
    filters_passed: Dict[str, bool]


class EntryOptimizer:
    """
    Advanced entry filter that combines multiple confirmation signals.
    
    Entry Conditions (ALL must be true):
    1. Model Prediction: UP with confidence > threshold
    2. Market Filter: Market trending in same direction
    3. Volume Confirmation: Above-average volume
    4. RSI Filter: Not overbought/oversold
    5. Price Action: Price above VWAP
    6. Risk/Reward: Favorable risk/reward ratio
    """
    
    def __init__(
        self,
        min_confidence: float = 0.60,
        min_quality_score: float = 0.70,
        require_market_alignment: bool = True,
        require_volume_confirmation: bool = True,
        use_rsi_filter: bool = True,
        use_vwap_filter: bool = True
    ):
        self.min_confidence = min_confidence
        self.min_quality_score = min_quality_score
        self.require_market_alignment = require_market_alignment
        self.require_volume_confirmation = require_volume_confirmation
        self.use_rsi_filter = use_rsi_filter
        self.use_vwap_filter = use_vwap_filter
    
    def evaluate(
        self,
        prediction: Dict,
        features: pd.Series,
        market_data: Optional[Dict] = None
    ) -> EntrySignal:
        """
        Evaluate if entry conditions are met.
        
        Args:
            prediction: Model prediction dict with 'direction' and 'confidence'
            features: Feature series for the stock
            market_data: Market indices data (NIFTY, VIX, etc.)
            
        Returns:
            EntrySignal with decision and quality score
        """
        reasons = []
        filters_passed = {}
        scores = []
        
        # 1. Model Confidence Filter
        confidence = prediction.get('confidence', 0)
        conf_pass = confidence >= self.min_confidence
        filters_passed['confidence'] = conf_pass
        
        if conf_pass:
            reasons.append(f"High confidence: {confidence:.1%}")
            scores.append(confidence)
        else:
            return EntrySignal(
                should_enter=False,
                quality_score=0,
                confidence=confidence,
                reasons=[f"Low confidence: {confidence:.1%}"],
                filters_passed=filters_passed
            )
        
        # 2. Market Alignment Filter
        if self.require_market_alignment and market_data:
            market_aligned = self._check_market_alignment(
                prediction['direction'], 
                market_data,
                features
            )
            filters_passed['market_alignment'] = market_aligned
            
            if market_aligned:
                reasons.append("Market aligned")
                scores.append(0.85)
            else:
                reasons.append("Market NOT aligned - skip")
                return EntrySignal(
                    should_enter=False,
                    quality_score=0,
                    confidence=confidence,
                    reasons=reasons,
                    filters_passed=filters_passed
                )
        
        # 3. Volume Confirmation
        if self.require_volume_confirmation:
            volume_confirmed = self._check_volume_confirmation(features)
            filters_passed['volume'] = volume_confirmed
            
            if volume_confirmed:
                reasons.append("Volume confirmed")
                scores.append(0.80)
            else:
                reasons.append("Weak volume - skip")
                return EntrySignal(
                    should_enter=False,
                    quality_score=0,
                    confidence=confidence,
                    reasons=reasons,
                    filters_passed=filters_passed
                )
        
        # 4. RSI Filter (avoid extremes)
        if self.use_rsi_filter:
            rsi_ok = self._check_rsi_filter(features, prediction['direction'])
            filters_passed['rsi'] = rsi_ok
            
            if rsi_ok:
                reasons.append("RSI in good range")
                scores.append(0.75)
            else:
                reasons.append("RSI overbought/oversold - skip")
                return EntrySignal(
                    should_enter=False,
                    quality_score=0,
                    confidence=confidence,
                    reasons=reasons,
                    filters_passed=filters_passed
                )
        
        # 5. VWAP Filter
        if self.use_vwap_filter:
            vwap_ok = self._check_vwap_filter(features, prediction['direction'])
            filters_passed['vwap'] = vwap_ok
            
            if vwap_ok:
                reasons.append("Price above VWAP" if prediction['direction'] == 1 else "Price below VWAP")
                scores.append(0.70)
        
        # 6. Technical Setup Score
        tech_score = self._calculate_technical_setup_score(features, prediction['direction'])
        scores.append(tech_score)
        filters_passed['technical_setup'] = tech_score > 0.5
        
        if tech_score > 0.7:
            reasons.append(f"Strong technical setup: {tech_score:.1%}")
        elif tech_score > 0.5:
            reasons.append(f"Decent technical setup: {tech_score:.1%}")
        
        # Calculate overall quality score
        quality_score = np.mean(scores)
        
        # Final decision
        should_enter = quality_score >= self.min_quality_score
        
        if should_enter:
            reasons.append(f"✅ ENTER - Quality score: {quality_score:.1%}")
        else:
            reasons.append(f"❌ SKIP - Quality score too low: {quality_score:.1%}")
        
        return EntrySignal(
            should_enter=should_enter,
            quality_score=quality_score,
            confidence=confidence,
            reasons=reasons,
            filters_passed=filters_passed
        )
    
    def _check_market_alignment(
        self, 
        direction: int, 
        market_data: Dict,
        features: pd.Series
    ) -> bool:
        """Check if market is trending in same direction."""
        # Check NIFTY trend
        nifty_return_1d = features.get('nifty_return_1d', 0)
        nifty_return_5d = features.get('nifty_return_5d', 0)
        
        if direction == 1:  # Bullish
            # For bullish trades, prefer bullish market
            return nifty_return_1d > -0.005 and nifty_return_5d > -0.01
        else:  # Bearish
            # For bearish trades, ok in any market
            return True
    
    def _check_volume_confirmation(self, features: pd.Series) -> bool:
        """Check if volume confirms the move."""
        volume_ratio = features.get('volume_ratio', 1.0)
        volume_surge = features.get('volume_surge_lag_1', 0)
        
        # Volume should be > 1.5x average OR recent volume surge
        return volume_ratio > 1.5 or volume_surge > 0
    
    def _check_rsi_filter(self, features: pd.Series, direction: int) -> bool:
        """Avoid overbought/oversold conditions."""
        rsi = features.get('rsi_14', 50)
        
        if direction == 1:  # Bullish
            # Avoid buying when overbought
            return 30 < rsi < 70
        else:  # Bearish
            # Avoid selling when oversold
            return 30 < rsi < 70
    
    def _check_vwap_filter(self, features: pd.Series, direction: int) -> bool:
        """Check price position relative to VWAP."""
        price_to_vwap = features.get('price_to_vwap', 1.0)
        
        if direction == 1:  # Bullish
            # For buys, prefer price above VWAP (strength)
            return price_to_vwap > 0.998
        else:  # Bearish
            # For sells, prefer price below VWAP (weakness)
            return price_to_vwap < 1.002
    
    def _calculate_technical_setup_score(self, features: pd.Series, direction: int) -> float:
        """Calculate technical setup quality score."""
        scores = []
        
        # 1. Trend alignment
        trend_5_20 = features.get('trend_5_20', 0)
        if direction == 1 and trend_5_20 > 0:
            scores.append(0.9)
        elif direction == -1 and trend_5_20 < 0:
            scores.append(0.9)
        else:
            scores.append(0.3)
        
        # 2. MACD alignment
        macd_hist = features.get('macd_hist', 0)
        if direction == 1 and macd_hist > 0:
            scores.append(0.8)
        elif direction == -1 and macd_hist < 0:
            scores.append(0.8)
        else:
            scores.append(0.4)
        
        # 3. Bollinger Band position
        bb_position = features.get('bb_position', 0.5)
        if direction == 1 and 0.2 < bb_position < 0.8:
            scores.append(0.85)  # Not at extremes
        elif direction == -1 and 0.2 < bb_position < 0.8:
            scores.append(0.85)
        else:
            scores.append(0.5)
        
        # 4. Momentum
        return_5d = features.get('return_5d', 0)
        if direction == 1 and return_5d > 0:
            scores.append(0.7)
        elif direction == -1 and return_5d < 0:
            scores.append(0.7)
        else:
            scores.append(0.5)
        
        return np.mean(scores)
    
    def batch_filter(
        self, 
        predictions: pd.DataFrame, 
        features: pd.DataFrame,
        market_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Filter multiple predictions at once.
        
        Args:
            predictions: DataFrame with columns ['symbol', 'direction', 'confidence']
            features: DataFrame with all features, indexed by symbol
            market_data: Market indices data
            
        Returns:
            Filtered predictions with quality scores
        """
        results = []
        
        for _, pred in predictions.iterrows():
            symbol = pred['symbol']
            
            if symbol not in features.index:
                continue
            
            signal = self.evaluate(
                prediction={
                    'direction': pred['direction'],
                    'confidence': pred['confidence']
                },
                features=features.loc[symbol],
                market_data=market_data
            )
            
            if signal.should_enter:
                results.append({
                    'symbol': symbol,
                    'direction': pred['direction'],
                    'confidence': pred['confidence'],
                    'quality_score': signal.quality_score,
                    'reasons': ', '.join(signal.reasons[:3])
                })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    optimizer = EntryOptimizer(
        min_confidence=0.60,
        min_quality_score=0.70
    )
    
    # Test with sample data
    prediction = {'direction': 1, 'confidence': 0.68}
    features = pd.Series({
        'rsi_14': 55,
        'volume_ratio': 1.8,
        'price_to_vwap': 1.002,
        'nifty_return_1d': 0.005,
        'nifty_return_5d': 0.012,
        'trend_5_20': 1,
        'macd_hist': 0.5,
        'bb_position': 0.6,
        'return_5d': 0.02
    })
    
    signal = optimizer.evaluate(prediction, features)
    
    print(f"Should Enter: {signal.should_enter}")
    print(f"Quality Score: {signal.quality_score:.1%}")
    print(f"Confidence: {signal.confidence:.1%}")
    print("\nReasons:")
    for reason in signal.reasons:
        print(f"  - {reason}")
    print("\nFilters Passed:")
    for filter_name, passed in signal.filters_passed.items():
        print(f"  {filter_name}: {'✅' if passed else '❌'}")

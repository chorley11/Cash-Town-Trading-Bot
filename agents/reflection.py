"""
Agent Reflection Module - Self-improvement through trade analysis

Each agent can:
1. Review its past trades and signals
2. Identify patterns in wins vs losses
3. Backtest parameter changes
4. Propose and apply improvements

This is the "learning" component of Gas Town agents.
"""
import json
import os
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .base import BaseStrategyAgent, Trade, Signal

logger = logging.getLogger(__name__)

@dataclass
class TradeAnalysis:
    """Analysis of a single trade"""
    trade_id: str
    symbol: str
    side: str
    pnl: float
    pnl_percent: float
    hold_time_hours: float
    entry_confidence: float
    exit_reason: str
    
    # What worked / didn't work
    signal_quality: str  # 'good', 'fair', 'poor'
    stop_loss_hit: bool
    take_profit_hit: bool
    premature_exit: bool
    late_entry: bool
    
    # Market context
    market_volatility: str  # 'low', 'medium', 'high'
    trend_alignment: bool
    volume_confirmed: bool
    
    lessons: List[str]

@dataclass
class ParameterSuggestion:
    """Suggested parameter change"""
    parameter: str
    current_value: Any
    suggested_value: Any
    reason: str
    expected_impact: str
    confidence: float  # 0-1

@dataclass
class ReflectionReport:
    """Full reflection report for an agent"""
    agent_id: str
    generated_at: str
    period_start: str
    period_end: str
    
    # Summary stats
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Patterns identified
    best_performing: Dict[str, Any]
    worst_performing: Dict[str, Any]
    patterns: List[str]
    
    # Recommendations
    parameter_suggestions: List[ParameterSuggestion]
    strategy_recommendations: List[str]
    
    # Overall assessment
    health_score: float  # 0-100
    improvement_priority: str  # 'low', 'medium', 'high', 'critical'

class AgentReflector:
    """
    Reflection engine for strategy agents.
    Analyzes past performance and suggests improvements.
    """
    
    def __init__(self, agent: BaseStrategyAgent, data_dir: str = None):
        self.agent = agent
        self.data_dir = data_dir or agent.data_dir
        self.reflection_dir = os.path.join(self.data_dir, 'reflections')
        os.makedirs(self.reflection_dir, exist_ok=True)
    
    def reflect(self, lookback_days: int = 7) -> ReflectionReport:
        """
        Perform full reflection on agent's recent performance.
        Returns a ReflectionReport with analysis and suggestions.
        """
        trades = self._get_recent_trades(lookback_days)
        
        if not trades:
            return self._empty_report(lookback_days)
        
        # Analyze trades
        analyses = [self._analyze_trade(t) for t in trades]
        
        # Calculate summary stats
        stats = self._calculate_stats(trades)
        
        # Identify patterns
        patterns = self._identify_patterns(analyses, trades)
        
        # Generate parameter suggestions
        suggestions = self._generate_suggestions(analyses, trades, stats)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analyses, stats, patterns)
        
        # Calculate health score
        health_score = self._calculate_health_score(stats, patterns)
        
        # Determine improvement priority
        priority = self._determine_priority(health_score, stats)
        
        report = ReflectionReport(
            agent_id=self.agent.agent_id,
            generated_at=datetime.utcnow().isoformat(),
            period_start=(datetime.utcnow() - timedelta(days=lookback_days)).isoformat(),
            period_end=datetime.utcnow().isoformat(),
            total_trades=len(trades),
            win_rate=stats['win_rate'],
            total_pnl=stats['total_pnl'],
            avg_win=stats['avg_win'],
            avg_loss=stats['avg_loss'],
            profit_factor=stats['profit_factor'],
            best_performing=stats['best'],
            worst_performing=stats['worst'],
            patterns=patterns,
            parameter_suggestions=suggestions,
            strategy_recommendations=recommendations,
            health_score=health_score,
            improvement_priority=priority
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _get_recent_trades(self, days: int) -> List[Trade]:
        """Get trades from the last N days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        return [t for t in self.agent.state.trades 
                if datetime.fromisoformat(str(t.exit_time)) > cutoff]
    
    def _analyze_trade(self, trade: Trade) -> TradeAnalysis:
        """Deep analysis of a single trade"""
        # Calculate hold time
        entry = datetime.fromisoformat(str(trade.entry_time))
        exit_time = datetime.fromisoformat(str(trade.exit_time))
        hold_hours = (exit_time - entry).total_seconds() / 3600
        
        # Determine signal quality based on outcome
        if trade.pnl > 0 and trade.pnl_percent > 2:
            signal_quality = 'good'
        elif trade.pnl > 0:
            signal_quality = 'fair'
        else:
            signal_quality = 'poor'
        
        # Analyze exit
        stop_loss_hit = 'stop' in trade.close_reason.lower()
        take_profit_hit = 'profit' in trade.close_reason.lower() or 'tp' in trade.close_reason.lower()
        
        # Check for premature/late issues
        premature_exit = trade.pnl > 0 and trade.pnl_percent < 1 and not take_profit_hit
        late_entry = trade.pnl < 0 and hold_hours < 1
        
        # Generate lessons
        lessons = []
        if stop_loss_hit and trade.pnl_percent < -3:
            lessons.append("Stop loss too wide - consider tightening")
        if take_profit_hit and trade.pnl_percent < 2:
            lessons.append("Take profit too tight - consider widening")
        if premature_exit:
            lessons.append("Exited too early - let winners run")
        if late_entry:
            lessons.append("Entered too late in move")
        if hold_hours > 48 and trade.pnl < 0:
            lessons.append("Held losing position too long")
        
        return TradeAnalysis(
            trade_id=trade.id,
            symbol=trade.symbol,
            side=trade.side,
            pnl=trade.pnl,
            pnl_percent=trade.pnl_percent,
            hold_time_hours=hold_hours,
            entry_confidence=0.6,  # Would need signal data
            exit_reason=trade.close_reason,
            signal_quality=signal_quality,
            stop_loss_hit=stop_loss_hit,
            take_profit_hit=take_profit_hit,
            premature_exit=premature_exit,
            late_entry=late_entry,
            market_volatility='medium',  # Would need market data
            trend_alignment=trade.pnl > 0,
            volume_confirmed=True,
            lessons=lessons
        )
    
    def _calculate_stats(self, trades: List[Trade]) -> Dict:
        """Calculate performance statistics"""
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Best/worst by symbol
        by_symbol = {}
        for t in trades:
            if t.symbol not in by_symbol:
                by_symbol[t.symbol] = {'trades': 0, 'pnl': 0}
            by_symbol[t.symbol]['trades'] += 1
            by_symbol[t.symbol]['pnl'] += t.pnl
        
        best_symbol = max(by_symbol.items(), key=lambda x: x[1]['pnl']) if by_symbol else (None, {})
        worst_symbol = min(by_symbol.items(), key=lambda x: x[1]['pnl']) if by_symbol else (None, {})
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'best': {'symbol': best_symbol[0], **best_symbol[1]} if best_symbol[0] else {},
            'worst': {'symbol': worst_symbol[0], **worst_symbol[1]} if worst_symbol[0] else {},
        }
    
    def _identify_patterns(self, analyses: List[TradeAnalysis], trades: List[Trade]) -> List[str]:
        """Identify patterns in trading behavior"""
        patterns = []
        
        # Stop loss pattern
        stop_losses = [a for a in analyses if a.stop_loss_hit]
        if len(stop_losses) / len(analyses) > 0.4:
            patterns.append("High stop-loss hit rate (>40%) - stops may be too tight")
        
        # Premature exit pattern
        premature = [a for a in analyses if a.premature_exit]
        if len(premature) / len(analyses) > 0.3:
            patterns.append("Frequent premature exits - consider trailing stops")
        
        # Time-based patterns
        short_holds = [a for a in analyses if a.hold_time_hours < 2 and a.pnl < 0]
        if len(short_holds) / len(analyses) > 0.3:
            patterns.append("Many quick losing trades - possible overtrading")
        
        # Win streak / lose streak
        consecutive_losses = 0
        max_losses = 0
        for t in trades:
            if t.pnl < 0:
                consecutive_losses += 1
                max_losses = max(max_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        if max_losses >= 5:
            patterns.append(f"Had {max_losses} consecutive losses - review entry conditions")
        
        # Symbol concentration
        symbols = [t.symbol for t in trades]
        if len(set(symbols)) < 3 and len(trades) > 10:
            patterns.append("Trading concentrated in few symbols - consider diversifying")
        
        return patterns
    
    def _generate_suggestions(self, analyses: List[TradeAnalysis], 
                             trades: List[Trade], stats: Dict) -> List[ParameterSuggestion]:
        """Generate parameter adjustment suggestions"""
        suggestions = []
        config = self.agent.config
        
        # Stop loss suggestions
        stop_losses = [a for a in analyses if a.stop_loss_hit]
        if len(stop_losses) / len(analyses) > 0.5:
            current_sl = config.get('stop_loss_atr', config.get('stop_loss_atr_mult', 2.0))
            suggestions.append(ParameterSuggestion(
                parameter='stop_loss_atr',
                current_value=current_sl,
                suggested_value=current_sl * 1.25,
                reason="High stop-loss hit rate suggests stops are too tight",
                expected_impact="Fewer stopped-out trades, potentially larger losses when wrong",
                confidence=0.7
            ))
        
        # Take profit suggestions
        if stats['avg_win'] < stats['avg_loss'] * 0.5:
            current_tp = config.get('take_profit_atr', config.get('take_profit_atr_mult', 3.0))
            suggestions.append(ParameterSuggestion(
                parameter='take_profit_atr',
                current_value=current_tp,
                suggested_value=current_tp * 1.5,
                reason="Average win is much smaller than average loss",
                expected_impact="Larger wins when right, potentially more trades stopped before TP",
                confidence=0.6
            ))
        
        # Confidence threshold
        if stats['win_rate'] < 40:
            current_conf = config.get('min_confidence', 0.55)
            suggestions.append(ParameterSuggestion(
                parameter='min_confidence',
                current_value=current_conf,
                suggested_value=min(current_conf + 0.1, 0.8),
                reason="Low win rate suggests taking too many low-quality signals",
                expected_impact="Fewer trades, but higher quality",
                confidence=0.65
            ))
        
        return suggestions
    
    def _generate_recommendations(self, analyses: List[TradeAnalysis], 
                                  stats: Dict, patterns: List[str]) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        if stats['win_rate'] < 40:
            recommendations.append("Focus on signal quality over quantity - be more selective")
        
        if stats['profit_factor'] < 1.0:
            recommendations.append("CRITICAL: Losing money overall - review entire strategy logic")
        elif stats['profit_factor'] < 1.5:
            recommendations.append("Profit factor below 1.5 - consider tightening entry criteria")
        
        if stats['total_trades'] < 5:
            recommendations.append("Insufficient data for reliable analysis - need more trades")
        
        # Based on patterns
        for pattern in patterns:
            if 'stop' in pattern.lower():
                recommendations.append("Review stop-loss methodology")
            if 'overtrading' in pattern.lower():
                recommendations.append("Add cooldown period between trades")
        
        return recommendations
    
    def _calculate_health_score(self, stats: Dict, patterns: List[str]) -> float:
        """Calculate overall strategy health score (0-100)"""
        score = 50  # Start neutral
        
        # Win rate impact
        if stats['win_rate'] >= 50:
            score += (stats['win_rate'] - 50) * 0.5
        else:
            score -= (50 - stats['win_rate']) * 0.5
        
        # Profit factor impact
        if stats['profit_factor'] >= 2.0:
            score += 20
        elif stats['profit_factor'] >= 1.5:
            score += 10
        elif stats['profit_factor'] >= 1.0:
            score += 0
        else:
            score -= 20
        
        # Pattern penalties
        score -= len(patterns) * 5
        
        return max(0, min(100, score))
    
    def _determine_priority(self, health_score: float, stats: Dict) -> str:
        """Determine improvement priority"""
        if stats['profit_factor'] < 1.0 or health_score < 30:
            return 'critical'
        elif health_score < 50:
            return 'high'
        elif health_score < 70:
            return 'medium'
        return 'low'
    
    def _empty_report(self, lookback_days: int) -> ReflectionReport:
        """Generate empty report when no trades"""
        return ReflectionReport(
            agent_id=self.agent.agent_id,
            generated_at=datetime.utcnow().isoformat(),
            period_start=(datetime.utcnow() - timedelta(days=lookback_days)).isoformat(),
            period_end=datetime.utcnow().isoformat(),
            total_trades=0,
            win_rate=0,
            total_pnl=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            best_performing={},
            worst_performing={},
            patterns=["Insufficient trades for analysis"],
            parameter_suggestions=[],
            strategy_recommendations=["Need more trades before making adjustments"],
            health_score=50,
            improvement_priority='low'
        )
    
    def _save_report(self, report: ReflectionReport):
        """Save reflection report to disk"""
        filename = f"reflection_{report.generated_at[:10]}.json"
        filepath = os.path.join(self.reflection_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        logger.info(f"Saved reflection report to {filepath}")
    
    def apply_suggestions(self, suggestions: List[ParameterSuggestion] = None, 
                         confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Apply parameter suggestions that meet confidence threshold.
        Returns dict of changes made.
        """
        if suggestions is None:
            # Get from most recent reflection
            report = self.reflect()
            suggestions = report.parameter_suggestions
        
        changes = {}
        for suggestion in suggestions:
            if suggestion.confidence >= confidence_threshold:
                old_value = self.agent.config.get(suggestion.parameter)
                self.agent.config[suggestion.parameter] = suggestion.suggested_value
                changes[suggestion.parameter] = {
                    'old': old_value,
                    'new': suggestion.suggested_value,
                    'reason': suggestion.reason
                }
                logger.info(f"Applied suggestion: {suggestion.parameter} "
                           f"{old_value} -> {suggestion.suggested_value}")
        
        # Save updated config
        if changes:
            self.agent._save_state()
        
        return changes

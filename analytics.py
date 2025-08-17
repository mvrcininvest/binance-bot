"""
Analytics Engine for Trading Bot v9.1
Provides performance analysis and statistics
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import func, and_, or_

from config import Config
from database import Session, Trade, SignalHistory, PerformanceMetrics

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Advanced analytics and performance tracking"""
    
    def __init__(self):
        """Initialize analytics engine"""
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        logger.info("Analytics Engine initialized")
    
    async def calculate_daily_stats(self) -> Dict:
        """Calculate daily trading statistics"""
        try:
            with Session() as session:
                today = datetime.utcnow().date()
                start_of_day = datetime.combine(today, datetime.min.time())
                
                # Get today's trades
                trades = session.query(Trade).filter(
                    and_(
                        Trade.created_at >= start_of_day,
                        Trade.status == 'closed'
                    )
                ).all()
                
                if not trades:
                    return self._empty_stats()
                
                # Calculate metrics
                total_trades = len(trades)
                winning_trades = [t for t in trades if t.realized_pnl > 0]
                losing_trades = [t for t in trades if t.realized_pnl < 0]
                
                total_pnl = sum(t.realized_pnl for t in trades)
                win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
                
                avg_win = np.mean([t.realized_pnl for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t.realized_pnl for t in losing_trades]) if losing_trades else 0
                
                # Profit factor
                gross_profit = sum(t.realized_pnl for t in winning_trades)
                gross_loss = abs(sum(t.realized_pnl for t in losing_trades))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Best and worst trades
                best_trade = max(trades, key=lambda t: t.realized_pnl).realized_pnl if trades else 0
                worst_trade = min(trades, key=lambda t: t.realized_pnl).realized_pnl if trades else 0
                
                # Average trade duration
                durations = []
                for trade in trades:
                    if trade.closed_at and trade.created_at:
                        duration = (trade.closed_at - trade.created_at).total_seconds() / 3600  # hours
                        durations.append(duration)
                avg_duration = np.mean(durations) if durations else 0
                
                # Store in database
                metrics = PerformanceMetrics(
                    period='daily',
                    period_start=start_of_day,
                    total_trades=total_trades,
                    winning_trades=len(winning_trades),
                    losing_trades=len(losing_trades),
                    win_rate=win_rate,
                    total_pnl=total_pnl,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    profit_factor=profit_factor,
                    best_trade=best_trade,
                    worst_trade=worst_trade,
                    avg_duration_hours=avg_duration
                )
                
                # Update or create
                existing = session.query(PerformanceMetrics).filter(
                    and_(
                        PerformanceMetrics.period == 'daily',
                        PerformanceMetrics.period_start == start_of_day
                    )
                ).first()
                
                if existing:
                    for key, value in metrics.__dict__.items():
                        if not key.startswith('_'):
                            setattr(existing, key, value)
                else:
                    session.add(metrics)
                
                session.commit()
                
                return {
                    'period': 'daily',
                    'date': today.isoformat(),
                    'total_trades': total_trades,
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'best_trade': best_trade,
                    'worst_trade': worst_trade,
                    'avg_duration_hours': avg_duration
                }
                
        except Exception as e:
            logger.error(f"Failed to calculate daily stats: {e}")
            return self._empty_stats()
    
    async def calculate_tier_performance(self) -> Dict:
        """Calculate performance by signal tier"""
        try:
            with Session() as session:
                # Get last 30 days of trades
                cutoff = datetime.utcnow() - timedelta(days=30)
                
                trades = session.query(Trade).filter(
                    and_(
                        Trade.created_at >= cutoff,
                        Trade.status == 'closed'
                    )
                ).all()
                
                # Group by tier
                tier_stats = defaultdict(lambda: {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'win_rate': 0,
                    'avg_leverage': 0,
                    'avg_strength': 0
                })
                
                for trade in trades:
                    tier = trade.signal_tier or 'Unknown'
                    tier_stats[tier]['trades'] += 1
                    
                    if trade.realized_pnl > 0:
                        tier_stats[tier]['wins'] += 1
                    elif trade.realized_pnl < 0:
                        tier_stats[tier]['losses'] += 1
                    
                    tier_stats[tier]['total_pnl'] += trade.realized_pnl
                    tier_stats[tier]['avg_leverage'] += trade.leverage_used or 0
                    tier_stats[tier]['avg_strength'] += trade.signal_strength or 0
                
                # Calculate averages
                for tier, stats in tier_stats.items():
                    if stats['trades'] > 0:
                        stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
                        stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
                        stats['avg_leverage'] = stats['avg_leverage'] / stats['trades']
                        stats['avg_strength'] = stats['avg_strength'] / stats['trades']
                
                # Rank tiers by profitability
                ranked_tiers = sorted(
                    tier_stats.items(),
                    key=lambda x: x[1]['total_pnl'],
                    reverse=True
                )
                
                return {
                    'period': '30_days',
                    'tiers': dict(tier_stats),
                    'ranking': [tier for tier, _ in ranked_tiers],
                    'best_tier': ranked_tiers[0][0] if ranked_tiers else None,
                    'worst_tier': ranked_tiers[-1][0] if ranked_tiers else None
                }
                
        except Exception as e:
            logger.error(f"Failed to calculate tier performance: {e}")
            return {'tiers': {}, 'ranking': []}
    
    async def calculate_symbol_performance(self) -> Dict:
        """Calculate performance by symbol"""
        try:
            with Session() as session:
                # Get last 30 days
                cutoff = datetime.utcnow() - timedelta(days=30)
                
                # Query aggregated stats by symbol
                results = session.query(
                    Trade.symbol,
                    func.count(Trade.id).label('total_trades'),
                    func.sum(Trade.realized_pnl).label('total_pnl'),
                    func.avg(Trade.realized_pnl).label('avg_pnl'),
                    func.sum(func.case([(Trade.realized_pnl > 0, 1)], else_=0)).label('wins'),
                    func.sum(func.case([(Trade.realized_pnl < 0, 1)], else_=0)).label('losses')
                ).filter(
                    and_(
                        Trade.created_at >= cutoff,
                        Trade.status == 'closed'
                    )
                ).group_by(Trade.symbol).all()
                
                symbol_stats = {}
                for row in results:
                    win_rate = (row.wins / row.total_trades * 100) if row.total_trades > 0 else 0
                    
                    symbol_stats[row.symbol] = {
                        'total_trades': row.total_trades,
                        'total_pnl': float(row.total_pnl or 0),
                        'avg_pnl': float(row.avg_pnl or 0),
                        'wins': row.wins or 0,
                        'losses': row.losses or 0,
                        'win_rate': win_rate
                    }
                
                # Find best and worst symbols
                if symbol_stats:
                    best_symbol = max(symbol_stats.items(), key=lambda x: x[1]['total_pnl'])
                    worst_symbol = min(symbol_stats.items(), key=lambda x: x[1]['total_pnl'])
                else:
                    best_symbol = worst_symbol = None
                
                return {
                    'period': '30_days',
                    'symbols': symbol_stats,
                    'best_symbol': best_symbol[0] if best_symbol else None,
                    'worst_symbol': worst_symbol[0] if worst_symbol else None,
                    'total_symbols_traded': len(symbol_stats)
                }
                
        except Exception as e:
            logger.error(f"Failed to calculate symbol performance: {e}")
            return {'symbols': {}}
    
    async def calculate_hourly_performance(self) -> Dict:
        """Calculate performance by hour of day"""
        try:
            with Session() as session:
                # Get last 30 days
                cutoff = datetime.utcnow() - timedelta(days=30)
                
                trades = session.query(Trade).filter(
                    and_(
                        Trade.created_at >= cutoff,
                        Trade.status == 'closed'
                    )
                ).all()
                
                # Group by hour
                hourly_stats = defaultdict(lambda: {
                    'trades': 0,
                    'wins': 0,
                    'total_pnl': 0,
                    'win_rate': 0
                })
                
                for trade in trades:
                    hour = trade.created_at.hour
                    hourly_stats[hour]['trades'] += 1
                    
                    if trade.realized_pnl > 0:
                        hourly_stats[hour]['wins'] += 1
                    
                    hourly_stats[hour]['total_pnl'] += trade.realized_pnl
                
                # Calculate win rates
                for hour, stats in hourly_stats.items():
                    if stats['trades'] > 0:
                        stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
                
                # Find best trading hours
                best_hours = sorted(
                    hourly_stats.items(),
                    key=lambda x: x[1]['total_pnl'],
                    reverse=True
                )[:3]
                
                return {
                    'hourly_performance': dict(hourly_stats),
                    'best_hours': [hour for hour, _ in best_hours],
                    'most_active_hour': max(hourly_stats.items(), key=lambda x: x[1]['trades'])[0] if hourly_stats else None
                }
                
        except Exception as e:
            logger.error(f"Failed to calculate hourly performance: {e}")
            return {'hourly_performance': {}}
    
    async def calculate_risk_metrics(self) -> Dict:
        """Calculate risk management metrics"""
        try:
            with Session() as session:
                # Get last 30 days
                cutoff = datetime.utcnow() - timedelta(days=30)
                
                trades = session.query(Trade).filter(
                    and_(
                        Trade.created_at >= cutoff,
                        Trade.status == 'closed'
                    )
                ).order_by(Trade.closed_at).all()
                
                if not trades:
                    return self._empty_risk_metrics()
                
                # Calculate returns series
                returns = [t.realized_pnl for t in trades]
                cumulative_returns = np.cumsum(returns)
                
                # Maximum drawdown
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - peak) / peak * 100
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                
                # Sharpe ratio (simplified - assuming 0 risk-free rate)
                if len(returns) > 1:
                    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                else:
                    sharpe_ratio = 0
                
                # Win/Loss streaks
                current_streak = 0
                max_win_streak = 0
                max_loss_streak = 0
                
                for trade in trades:
                    if trade.realized_pnl > 0:
                        if current_streak >= 0:
                            current_streak += 1
                            max_win_streak = max(max_win_streak, current_streak)
                        else:
                            current_streak = 1
                    elif trade.realized_pnl < 0:
                        if current_streak <= 0:
                            current_streak -= 1
                            max_loss_streak = max(max_loss_streak, abs(current_streak))
                        else:
                            current_streak = -1
                
                # Risk/Reward ratio
                wins = [t.realized_pnl for t in trades if t.realized_pnl > 0]
                losses = [abs(t.realized_pnl) for t in trades if t.realized_pnl < 0]
                
                avg_win = np.mean(wins) if wins else 0
                avg_loss = np.mean(losses) if losses else 0
                risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
                
                # Recovery factor
                total_pnl = sum(returns)
                recovery_factor = total_pnl / abs(max_drawdown) if max_drawdown < 0 else float('inf')
                
                return {
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'max_win_streak': max_win_streak,
                    'max_loss_streak': max_loss_streak,
                    'risk_reward_ratio': risk_reward_ratio,
                    'recovery_factor': recovery_factor,
                    'current_streak': current_streak
                }
                
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            return self._empty_risk_metrics()
    
    async def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        try:
            # Gather all analytics
            daily_stats = await self.calculate_daily_stats()
            tier_performance = await self.calculate_tier_performance()
            symbol_performance = await self.calculate_symbol_performance()
            hourly_performance = await self.calculate_hourly_performance()
            risk_metrics = await self.calculate_risk_metrics()
            
            # Compile report
            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'daily_stats': daily_stats,
                'tier_performance': tier_performance,
                'symbol_performance': symbol_performance,
                'hourly_performance': hourly_performance,
                'risk_metrics': risk_metrics,
                'recommendations': self._generate_recommendations(
                    daily_stats, tier_performance, symbol_performance, risk_metrics
                )
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {}
    
    def _generate_recommendations(self, daily_stats: Dict, tier_perf: Dict, 
                                 symbol_perf: Dict, risk_metrics: Dict) -> List[str]:
        """Generate trading recommendations based on analytics"""
        recommendations = []
        
        # Based on win rate
        if daily_stats.get('win_rate', 0) < 40:
            recommendations.append("Consider switching to Conservative mode - low win rate detected")
        elif daily_stats.get('win_rate', 0) > 70:
            recommendations.append("Performance is strong - consider Aggressive mode for higher returns")
        
        # Based on tier performance
        if tier_perf.get('best_tier'):
            recommendations.append(f"Focus on {tier_perf['best_tier']} tier signals - best performance")
        
        if tier_perf.get('worst_tier'):
            worst = tier_perf['worst_tier']
            if tier_perf['tiers'].get(worst, {}).get('total_pnl', 0) < 0:
                recommendations.append(f"Consider avoiding {worst} tier signals - negative performance")
        
        # Based on risk metrics
        if risk_metrics.get('max_drawdown', 0) < -20:
            recommendations.append("High drawdown detected - consider reducing position sizes")
        
        if risk_metrics.get('max_loss_streak', 0) > 5:
            recommendations.append("Long losing streak detected - review entry criteria")
        
        # Based on symbols
        if symbol_perf.get('worst_symbol'):
            worst_symbol = symbol_perf['worst_symbol']
            if symbol_perf['symbols'].get(worst_symbol, {}).get('total_pnl', 0) < 0:
                recommendations.append(f"Consider blacklisting {worst_symbol} - consistent losses")
        
        # Based on hourly performance
        if daily_stats.get('total_trades', 0) > 20:
            recommendations.append("High trading frequency - consider being more selective")
        
        return recommendations
    
    def _empty_stats(self) -> Dict:
        """Return empty stats structure"""
        return {
            'period': 'daily',
            'date': datetime.utcnow().date().isoformat(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'avg_duration_hours': 0
        }
    
    def _empty_risk_metrics(self) -> Dict:
        """Return empty risk metrics"""
        return {
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'risk_reward_ratio': 0,
            'recovery_factor': 0,
            'current_streak': 0
        }


# Singleton instance
analytics_engine = AnalyticsEngine()
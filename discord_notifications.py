"""
Discord notifications system for Trading Bot v9.1
Enhanced with tier-based notifications and rich embeds
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import aiohttp
from discord_webhook import AsyncDiscordWebhook, DiscordEmbed

from config import Config
from database import Session, get_performance_stats, get_tier_performance

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Enhanced Discord notification system"""
    
    def __init__(self):
        """Initialize Discord notifier"""
        self.webhook_url = Config.DISCORD_WEBHOOK
        self.alerts_webhook = Config.DISCORD_ALERTS_WEBHOOK
        self.trades_webhook = Config.DISCORD_TRADES_WEBHOOK
        self.errors_webhook = Config.DISCORD_ERRORS_WEBHOOK
        
        # Tier colors for embeds
        self.tier_colors = {
            'Platinum': 0xE5E4E2,  # Platinum
            'Premium': 0xFFD700,   # Gold
            'Standard': 0x0099FF,  # Blue
            'Quick': 0x00FF00,     # Green
            'Emergency': 0xFF0000  # Red
        }
        
        # Action emojis
        self.action_emojis = {
            'BUY': '🟢',
            'SELL': '🔴',
            'LONG': '📈',
            'SHORT': '📉',
            'CLOSE': '⏹️'
        }
        
        # Status emojis
        self.status_emojis = {
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'info': 'ℹ️',
            'profit': '💰',
            'loss': '📉'
        }
        
        logger.info("Discord notifier initialized")
    
    async def _send_webhook(self, webhook_url: str, embed: DiscordEmbed, username: str = "Trading Bot v9.1") -> bool:
        """Send webhook with embed"""
        if not webhook_url:
            return False
        
        try:
            webhook = AsyncDiscordWebhook(
                url=webhook_url,
                username=username,
                rate_limit_retry=True
            )
            webhook.add_embed(embed)
            
            response = await webhook.execute()
            return response.status_code in [200, 204]
            
        except Exception as e:
            logger.error(f"Discord webhook error: {e}")
            return False
    
    async def send_signal_received_notification(self, alert: Dict) -> None:
        """Send notification when signal is received"""
        try:
            tier = alert.get('tier', 'Standard')
            symbol = alert.get('symbol', 'UNKNOWN')
            action = alert.get('action', 'UNKNOWN')
            strength = alert.get('strength', 0)
            
            # Create embed
            embed = DiscordEmbed(
                title=f"{self.action_emojis.get(action, '📊')} Signal Received - {tier}",
                description=f"**{symbol}** - {action}",
                color=self.tier_colors.get(tier, 0x808080),
                timestamp=datetime.utcnow()
            )
            
            # Add fields
            embed.add_embed_field(name="Tier", value=tier, inline=True)
            embed.add_embed_field(name="Strength", value=f"{strength:.1f}%", inline=True)
            embed.add_embed_field(name="Action", value=action, inline=True)
            
            # Price info
            embed.add_embed_field(name="Price", value=f"${alert.get('price', 0):.2f}", inline=True)
            if alert.get('stop_loss'):
                embed.add_embed_field(name="Stop Loss", value=f"${alert['stop_loss']:.2f}", inline=True)
            if alert.get('take_profit_1'):
                embed.add_embed_field(name="TP1", value=f"${alert['take_profit_1']:.2f}", inline=True)
            
            # v9.1 Enhanced info
            if alert.get('institutional_flow', 0) > 0.5:
                embed.add_embed_field(
                    name="📊 Institutional Flow",
                    value=f"{alert['institutional_flow']:.2%}",
                    inline=False
                )
            
            if alert.get('fake_breakout_detected'):
                embed.add_embed_field(
                    name="⚠️ Warning",
                    value="Fake breakout detected",
                    inline=False
                )
            
            # Regime info
            regime = alert.get('regime', 'NEUTRAL')
            regime_emoji = {'BULLISH': '🐂', 'BEARISH': '🐻', 'NEUTRAL': '➖'}.get(regime, '❓')
            embed.add_embed_field(
                name="Market Regime",
                value=f"{regime_emoji} {regime}",
                inline=True
            )
            
            # Send to alerts channel
            await self._send_webhook(self.alerts_webhook or self.webhook_url, embed)
            
        except Exception as e:
            logger.error(f"Failed to send signal notification: {e}")
    
    async def send_signal_decision_notification(
        self,
        alert: Dict,
        accepted: bool,
        reason: str,
        trade_id: Optional[str] = None
    ) -> None:
        """Send notification about signal decision"""
        try:
            tier = alert.get('tier', 'Standard')
            symbol = alert.get('symbol', 'UNKNOWN')
            action = alert.get('action', 'UNKNOWN')
            
            # Choose color based on decision
            color = 0x00FF00 if accepted else 0xFF0000
            
            # Create embed
            embed = DiscordEmbed(
                title=f"{self.status_emojis['success' if accepted else 'error']} Signal {('Accepted' if accepted else 'Rejected')}",
                description=f"**{symbol}** - {action} ({tier})",
                color=color,
                timestamp=datetime.utcnow()
            )
            
            # Add decision reason
            embed.add_embed_field(
                name="Decision",
                value=reason,
                inline=False
            )
            
            if accepted and trade_id:
                embed.add_embed_field(name="Trade ID", value=trade_id, inline=True)
                
                # Add execution details
                if alert.get('leverage'):
                    embed.add_embed_field(name="Leverage", value=f"{alert['leverage']}x", inline=True)
                if alert.get('position_size'):
                    embed.add_embed_field(name="Size", value=f"${alert['position_size']:.2f}", inline=True)
            
            # Send to appropriate channel
            webhook_url = self.trades_webhook if accepted else self.alerts_webhook
            await self._send_webhook(webhook_url or self.webhook_url, embed)
            
        except Exception as e:
            logger.error(f"Failed to send decision notification: {e}")
    
    async def send_trade_opened_notification(self, trade_data: Dict) -> None:
        """Send notification when trade is opened"""
        try:
            symbol = trade_data.get('symbol', 'UNKNOWN')
            side = trade_data.get('side', 'UNKNOWN')
            tier = trade_data.get('tier', 'Standard')
            
            # Create embed
            embed = DiscordEmbed(
                title=f"{self.action_emojis.get(side, '📊')} Position Opened",
                description=f"**{symbol}** - {side}",
                color=self.tier_colors.get(tier, 0x808080),
                timestamp=datetime.utcnow()
            )
            
            # Position details
            embed.add_embed_field(name="Entry Price", value=f"${trade_data.get('entry_price', 0):.2f}", inline=True)
            embed.add_embed_field(name="Position Size", value=f"${trade_data.get('position_size', 0):.2f}", inline=True)
            embed.add_embed_field(name="Leverage", value=f"{trade_data.get('leverage', 1)}x", inline=True)
            
            # Risk management
            if trade_data.get('stop_loss'):
                risk_amount = abs(trade_data['position_size'] * 
                                 (trade_data['stop_loss'] - trade_data['entry_price']) / 
                                 trade_data['entry_price'])
                embed.add_embed_field(name="Stop Loss", value=f"${trade_data['stop_loss']:.2f}", inline=True)
                embed.add_embed_field(name="Risk", value=f"${risk_amount:.2f}", inline=True)
            
            # Take profits
            if trade_data.get('take_profit_1'):
                embed.add_embed_field(name="TP1", value=f"${trade_data['take_profit_1']:.2f}", inline=True)
            if trade_data.get('take_profit_2'):
                embed.add_embed_field(name="TP2", value=f"${trade_data['take_profit_2']:.2f}", inline=True)
            if trade_data.get('take_profit_3'):
                embed.add_embed_field(name="TP3", value=f"${trade_data['take_profit_3']:.2f}", inline=True)
            
            # Tier and signal info
            embed.add_embed_field(name="Signal Tier", value=tier, inline=True)
            if trade_data.get('signal_strength'):
                embed.add_embed_field(name="Signal Strength", value=f"{trade_data['signal_strength']:.1f}%", inline=True)
            
            # Send to trades channel
            await self._send_webhook(self.trades_webhook or self.webhook_url, embed)
            
        except Exception as e:
            logger.error(f"Failed to send trade opened notification: {e}")
    
    async def send_trade_closed_notification(self, trade_data: Dict) -> None:
        """Send notification when trade is closed"""
        try:
            symbol = trade_data.get('symbol', 'UNKNOWN')
            pnl = trade_data.get('realized_pnl', 0)
            pnl_percent = trade_data.get('pnl_percent', 0)
            
            # Determine color and emoji based on PnL
            if pnl > 0:
                color = 0x00FF00
                emoji = self.status_emojis['profit']
                title = "Position Closed - PROFIT"
            elif pnl < 0:
                color = 0xFF0000
                emoji = self.status_emojis['loss']
                title = "Position Closed - LOSS"
            else:
                color = 0x808080
                emoji = '➖'
                title = "Position Closed - BREAKEVEN"
            
            # Create embed
            embed = DiscordEmbed(
                title=f"{emoji} {title}",
                description=f"**{symbol}**",
                color=color,
                timestamp=datetime.utcnow()
            )
            
            # Trade details
            embed.add_embed_field(name="Entry Price", value=f"${trade_data.get('entry_price', 0):.2f}", inline=True)
            embed.add_embed_field(name="Exit Price", value=f"${trade_data.get('exit_price', 0):.2f}", inline=True)
            embed.add_embed_field(name="Position Size", value=f"${trade_data.get('position_size', 0):.2f}", inline=True)
            
            # PnL
            embed.add_embed_field(name="Realized PnL", value=f"${pnl:.2f}", inline=True)
            embed.add_embed_field(name="PnL %", value=f"{pnl_percent:.2f}%", inline=True)
            embed.add_embed_field(name="Close Reason", value=trade_data.get('close_reason', 'Manual'), inline=True)
            
            # Trade duration
            if trade_data.get('duration'):
                embed.add_embed_field(name="Duration", value=trade_data['duration'], inline=True)
            
            # Tier info
            if trade_data.get('tier'):
                embed.add_embed_field(name="Signal Tier", value=trade_data['tier'], inline=True)
            
            # Send to trades channel
            await self._send_webhook(self.trades_webhook or self.webhook_url, embed)
            
        except Exception as e:
            logger.error(f"Failed to send trade closed notification: {e}")
    
    async def send_performance_report(self, stats: Dict) -> None:
        """Send daily/weekly performance report"""
        try:
            period = stats.get('period', 'daily')
            total_trades = stats.get('total_trades', 0)
            win_rate = stats.get('win_rate', 0)
            total_pnl = stats.get('total_pnl', 0)
            
            # Determine color based on PnL
            color = 0x00FF00 if total_pnl > 0 else 0xFF0000 if total_pnl < 0 else 0x808080
            
            # Create embed
            embed = DiscordEmbed(
                title=f"📊 {period.capitalize()} Performance Report",
                description=f"Performance summary for {datetime.utcnow().strftime('%Y-%m-%d')}",
                color=color,
                timestamp=datetime.utcnow()
            )
            
            # Overall stats
            embed.add_embed_field(name="Total Trades", value=str(total_trades), inline=True)
            embed.add_embed_field(name="Win Rate", value=f"{win_rate:.1f}%", inline=True)
            embed.add_embed_field(name="Total PnL", value=f"${total_pnl:.2f}", inline=True)
            
            # Detailed stats
            if stats.get('winning_trades'):
                embed.add_embed_field(name="Winning Trades", value=str(stats['winning_trades']), inline=True)
            if stats.get('losing_trades'):
                embed.add_embed_field(name="Losing Trades", value=str(stats['losing_trades']), inline=True)
            if stats.get('avg_win'):
                embed.add_embed_field(name="Avg Win", value=f"${stats['avg_win']:.2f}", inline=True)
            if stats.get('avg_loss'):
                embed.add_embed_field(name="Avg Loss", value=f"${stats['avg_loss']:.2f}", inline=True)
            
            # Tier breakdown
            if stats.get('tier_performance'):
                tier_text = []
                for tier, tier_stats in stats['tier_performance'].items():
                    tier_text.append(f"**{tier}**: {tier_stats['trades']} trades, {tier_stats['win_rate']:.1f}% WR, ${tier_stats['pnl']:.2f}")
                
                if tier_text:
                    embed.add_embed_field(
                        name="Performance by Tier",
                        value="\n".join(tier_text),
                        inline=False
                    )
            
            # Best and worst trades
            if stats.get('best_trade'):
                embed.add_embed_field(name="Best Trade", value=f"${stats['best_trade']:.2f}", inline=True)
            if stats.get('worst_trade'):
                embed.add_embed_field(name="Worst Trade", value=f"${stats['worst_trade']:.2f}", inline=True)
            
            # Send to main webhook
            await self._send_webhook(self.webhook_url, embed)
            
        except Exception as e:
            logger.error(f"Failed to send performance report: {e}")
    
    async def send_error_notification(self, error_message: str, critical: bool = False) -> None:
        """Send error notification"""
        try:
            # Create embed
            embed = DiscordEmbed(
                title=f"{self.status_emojis['error']} {'CRITICAL ERROR' if critical else 'Error'}",
                description=error_message[:1024],  # Discord limit
                color=0xFF0000,
                timestamp=datetime.utcnow()
            )
            
            # Add context
            embed.add_embed_field(name="Time", value=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'), inline=True)
            embed.add_embed_field(name="Severity", value="CRITICAL" if critical else "ERROR", inline=True)
            
            # Send to errors channel or main
            await self._send_webhook(self.errors_webhook or self.webhook_url, embed)
            
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
    
    async def send_alert_notification(self, message: str, alert_type: str = "info") -> None:
        """Send general alert notification"""
        try:
            # Choose color and emoji based on type
            colors = {
                'info': 0x0099FF,
                'warning': 0xFFFF00,
                'success': 0x00FF00,
                'error': 0xFF0000
            }
            
            # Create embed
            embed = DiscordEmbed(
                title=f"{self.status_emojis.get(alert_type, 'ℹ️')} Alert",
                description=message,
                color=colors.get(alert_type, 0x808080),
                timestamp=datetime.utcnow()
            )
            
            # Send to alerts channel
            await self._send_webhook(self.alerts_webhook or self.webhook_url, embed)
            
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
    
    async def send_startup_notification(self) -> None:
        """Send bot startup notification"""
        try:
            embed = DiscordEmbed(
                title="🚀 Bot Started",
                description="Trading Bot v9.1 is now online",
                color=0x00FF00,
                timestamp=datetime.utcnow()
            )
            
            # Add system info
            embed.add_embed_field(name="Version", value="9.1", inline=True)
            embed.add_embed_field(name="Mode", value=Config.DEFAULT_MODE, inline=True)
            embed.add_embed_field(name="Indicator", value="v9.1", inline=True)
            
            # Add configuration info
            embed.add_embed_field(
                name="Configuration",
                value=f"Max Positions: {Config.MAX_POSITIONS}\n"
                      f"Risk per Trade: {Config.RISK_PER_TRADE}%\n"
                      f"Emergency Threshold: {Config.EMERGENCY_CLOSE_THRESHOLD}%",
                inline=False
            )
            
            await self._send_webhook(self.webhook_url, embed)
            
        except Exception as e:
            logger.error(f"Failed to send startup notification: {e}")
    
    async def send_shutdown_notification(self) -> None:
        """Send bot shutdown notification"""
        try:
            embed = DiscordEmbed(
                title="🛑 Bot Stopped",
                description="Trading Bot v9.1 is shutting down",
                color=0xFF0000,
                timestamp=datetime.utcnow()
            )
            
            # Add shutdown stats if available
            with Session() as session:
                stats = get_performance_stats(session, "daily")
                if stats['total_trades'] > 0:
                    embed.add_embed_field(name="Today's Trades", value=str(stats['total_trades']), inline=True)
                    embed.add_embed_field(name="Today's PnL", value=f"${stats['total_pnl']:.2f}", inline=True)
            
            await self._send_webhook(self.webhook_url, embed)
            
        except Exception as e:
            logger.error(f"Failed to send shutdown notification: {e}")
    
    async def send_control_notification(self, message: str) -> None:
        """Send control action notification (pause/resume/mode change)"""
        try:
            embed = DiscordEmbed(
                title="⚙️ Control Action",
                description=message,
                color=0xFFFF00,
                timestamp=datetime.utcnow()
            )
            
            await self._send_webhook(self.webhook_url, embed)
            
        except Exception as e:
            logger.error(f"Failed to send control notification: {e}")
    
    async def send_emergency_notification(self, reason: str, positions_closed: int) -> None:
        """Send emergency mode notification"""
        try:
            embed = DiscordEmbed(
                title="🚨 EMERGENCY MODE ACTIVATED",
                description=f"**Reason:** {reason}",
                color=0xFF0000,
                timestamp=datetime.utcnow()
            )
            
            embed.add_embed_field(name="Positions Closed", value=str(positions_closed), inline=True)
            embed.add_embed_field(name="Action", value="All positions closed", inline=True)
            
            # Send to all channels
            await self._send_webhook(self.webhook_url, embed)
            if self.errors_webhook:
                await self._send_webhook(self.errors_webhook, embed)
            
        except Exception as e:
            logger.error(f"Failed to send emergency notification: {e}")


# Singleton instance
discord_notifier = DiscordNotifier()
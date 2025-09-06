"""
Discord notifications system for Trading Bot v9.1
Enhanced with comprehensive v9.1 features, rich embeds, and advanced analytics
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import aiohttp
from discord_webhook import AsyncDiscordWebhook, DiscordEmbed

from config import Config
from database import Session,get_performance_stats,get_tier_performance
from database import (
    Session, get_performance_stats, get_tier_performance,
    log_execution_trace, complete_execution_trace
)

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Enhanced Discord notification system v9.1"""

    def __init__(self):
        """Initialize Discord notifier with v9.1 enhancements"""
        self.webhook_url = Config.DISCORD_WEBHOOK
        self.alerts_webhook = Config.DISCORD_ALERTS_WEBHOOK
        self.trades_webhook = Config.DISCORD_TRADES_WEBHOOK
        self.errors_webhook = Config.DISCORD_ERRORS_WEBHOOK
        self.analytics_webhook = Config.DISCORD_ANALYTICS_WEBHOOK

        # v9.1 CORE: Enhanced tier colors
        self.tier_colors = {
            "Emergency": 0xFF0000,  # Red - Highest priority
            "Platinum": 0xE5E4E2,  # Platinum
            "Premium": 0xFFD700,  # Gold
            "Standard": 0x0099FF,  # Blue
            "Quick": 0x00FF00,  # Green
        }

        # v9.1 CORE: Enhanced action emojis
        self.action_emojis = {
            "buy": "🟢",
            "sell": "🔴",
            "long": "📈",
            "short": "📉",
            "close": "⏹️",
            "emergency_close": "🚨",
        }

        # v9.1 CORE: Enhanced status emojis
        self.status_emojis = {
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "info": "ℹ️",
            "profit": "💰",
            "loss": "📉",
            "breakeven": "➖",
            "emergency": "🚨",
            "paused": "⏸️",
            "resumed": "▶️",
            "mode_change": "⚙️",
        }

        # v9.1 NEW: Market regime emojis
        self.regime_emojis = {
            "TRENDING_UP": "🐂",
            "TRENDING_DOWN": "🐻",
            "RANGING": "↔️",
            "VOLATILE": "⚡",
            "BREAKOUT": "🚀",
            "NEUTRAL": "➖",
        }

        # v9.1 NEW: Context emojis
        self.context_emojis = {
            "institutional_flow": "🏦",
            "fake_breakout": "⚠️",
            "retest": "🔄",
            "liquidity_grab": "💧",
            "order_block": "📦",
            "fair_value_gap": "📊",
            "volume_spike": "📈",
        }

        logger.info("Discord notifier v9.1 initialized with enhanced features")

    async def _send_webhook(
        self, webhook_url: str, embed: DiscordEmbed, username: str = "Trading Bot v9.1"
    ) -> bool:
        """Send webhook with embed - v9.1 ENHANCED"""
        start_time = datetime.utcnow()

        if not webhook_url:
            logger.error(f"❌ DiscordNotifier: webhook_url jest pusty/None dla '{username}' - sprawdź .env: DISCORD_WEBHOOK, DISCORD_ALERTS_WEBHOOK, DISCORD_TRADES_WEBHOOK, DISCORD_ERRORS_WEBHOOK")
            await self.log_notification_trace("send_webhook", False, 0, f"Missing URL for {username}")
            return False

        try:
            webhook = AsyncDiscordWebhook(
                url=webhook_url, username=username, rate_limit_retry=True
            )
            webhook.add_embed(embed)

            response = await webhook.execute()
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            success = response.status_code in [200, 204]

            await self.log_notification_trace("send_webhook", success, duration, f"Status {response.status_code}")
            return success

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"Discord webhook error: {e}")
            await self.log_notification_trace("send_webhook", False, duration, str(e))
            return False

    async def send_signal_notification(
        self,
        alert: Dict,
        accepted: bool,
        reason: str,
        trade_id: Optional[str] = None,
        analysis_data: Optional[Dict] = None,
    ) -> None:
        """Send comprehensive signal notification - v9.1 ENHANCED"""
        try:
            start_time = datetime.utcnow()
            symbol = alert.get("symbol", "UNKNOWN")
            tier = alert.get("tier", "Standard")
            tier = alert.get("tier", "Standard")
            symbol = alert.get("symbol", "UNKNOWN")
            action = alert.get("action", "unknown")
            strength = alert.get("strength", 0)

            # Choose color and title based on acceptance
            if accepted:
                color = self.tier_colors.get(tier, 0x808080)
                title = (
                    f"{self.action_emojis.get(action, '📊')} Signal ACCEPTED - {tier}"
                )
                status_emoji = self.status_emojis["success"]
            else:
                color = 0xFF6B6B  # Light red for rejection
                title = f"{self.status_emojis['warning']} Signal REJECTED - {tier}"
                status_emoji = self.status_emojis["warning"]

            # Create enhanced embed
            embed = DiscordEmbed(
                title=f"{status_emoji} {title}",
                description=f"**{symbol}** - {action.upper()}",
                color=color,
                timestamp=datetime.utcnow(),
            )

            # v9.1 CORE: Basic signal info
            embed.add_embed_field(name="Tier", value=tier, inline=True)
            embed.add_embed_field(name="Strength", value=f"{strength:.3f}", inline=True)
            embed.add_embed_field(name="Action", value=action.upper(), inline=True)

            # Price information
            if alert.get("price"):
                embed.add_embed_field(
                    name="Price", value=f"${alert['price']:.4f}", inline=True
                )
            if alert.get("sl"):
                embed.add_embed_field(
                    name="Stop Loss", value=f"${alert['sl']:.4f}", inline=True
                )
            if alert.get("tp1"):
                embed.add_embed_field(
                    name="TP1", value=f"${alert['tp1']:.4f}", inline=True
                )

            # v9.1 CORE: Enhanced market context
            if alert.get("enhanced_regime"):
                regime = alert["enhanced_regime"]
                regime_emoji = self.regime_emojis.get(regime, "❓")
                confidence = alert.get("regime_confidence", 0)
                embed.add_embed_field(
                    name="Market Regime",
                    value=f"{regime_emoji} {regime} ({confidence:.1%})",
                    inline=True,
                )

            # v9.1 CORE: Institutional flow
            institutional_flow = alert.get("institutional_flow", 0)
            if institutional_flow != 0:
                flow_direction = "Buying" if institutional_flow > 0 else "Selling"
                embed.add_embed_field(
                    name=f"{self.context_emojis['institutional_flow']} Institutional Flow",
                    value=f"{flow_direction} ({abs(institutional_flow):.1%})",
                    inline=True,
                )

            # v9.1 CORE: Multi-timeframe agreement
            mtf_agreement = alert.get("mtf_agreement_ratio", 0)
            if mtf_agreement > 0:
                embed.add_embed_field(
                    name="MTF Agreement", value=f"{mtf_agreement:.1%}", inline=True
                )

            # v9.1 CORE: Volume context
            if alert.get("volume_spike"):
                volume_ratio = alert.get("volume_ratio", 1.0)
                embed.add_embed_field(
                    name=f"{self.context_emojis['volume_spike']} Volume Spike",
                    value=f"{volume_ratio:.1f}x average",
                    inline=True,
                )

            # v9.1 CORE: Enhanced context flags
            context_flags = []
            if alert.get("fake_breakout_detected"):
                penalty = alert.get("fake_breakout_penalty", 1.0)
                context_flags.append(
                    f"{self.context_emojis['fake_breakout']} Fake Breakout (Penalty: {penalty:.2f})"
                )

            if alert.get("order_block_retest"):
                context_flags.append(
                    f"{self.context_emojis['order_block']} Order Block Retest"
                )

            if alert.get("liquidity_grab"):
                context_flags.append(
                    f"{self.context_emojis['liquidity_grab']} Liquidity Grab"
                )

            if alert.get("fair_value_gap"):
                context_flags.append(
                    f"{self.context_emojis['fair_value_gap']} Fair Value Gap"
                )

            if context_flags:
                embed.add_embed_field(
                    name="Context Signals", value="\n".join(context_flags), inline=False
                )

            # Decision reason
            embed.add_embed_field(
                name="Decision Reason",
                value=reason[:1024],  # Discord limit
                inline=False,
            )

            # v9.1 CORE: Trade execution details (if accepted)
            if accepted:
                if trade_id:
                    embed.add_embed_field(name="Trade ID", value=trade_id, inline=True)

                if alert.get("leverage"):
                    embed.add_embed_field(
                        name="Leverage", value=f"{alert['leverage']}x", inline=True
                    )

                if analysis_data and analysis_data.get("trade_parameters"):
                    params = analysis_data["trade_parameters"]
                    multiplier = params.get("position_multiplier", 1.0)
                    embed.add_embed_field(
                        name="Size Multiplier", value=f"{multiplier:.2f}x", inline=True
                    )

            # v9.1 CORE: Technical indicators summary
            tech_indicators = []
            if alert.get("rsi"):
                tech_indicators.append(f"RSI: {alert['rsi']:.1f}")
            if alert.get("mfi"):
                tech_indicators.append(f"MFI: {alert['mfi']:.1f}")
            if alert.get("adx"):
                tech_indicators.append(f"ADX: {alert['adx']:.1f}")

            if tech_indicators:
                embed.add_embed_field(
                    name="Technical Indicators",
                    value=" | ".join(tech_indicators),
                    inline=False,
                )

            # v9.1 CORE: Processing metadata
            if analysis_data:
                processing_time = analysis_data.get("processing_time_ms", 0)
                embed.add_embed_field(
                    name="Processing Time", value=f"{processing_time}ms", inline=True
                )

            # Send to appropriate channel
            webhook_url = self.trades_webhook if accepted else self.alerts_webhook
            result = await self._send_webhook(webhook_url or self.webhook_url, embed)
    
            # Log wrapper execution
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            status = "ACCEPTED" if accepted else "REJECTED"
            await self.log_notification_trace(
                f"signal_notification_{status.lower()}", 
                result, 
                duration, 
                f"{symbol} {tier} - {reason[:100]}"
            )

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("signal_notification", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send signal notification: {e}")

    async def send_trade_opened_notification(self, trade_data: Dict) -> None:
        """Send enhanced trade opened notification - v9.1 ENHANCED"""
        try:
            start_time = datetime.utcnow()
            symbol = trade_data.get("symbol", "UNKNOWN")
            side = trade_data.get("side", "UNKNOWN")
            symbol = trade_data.get("symbol", "UNKNOWN")
            side = trade_data.get("side", "UNKNOWN")
            tier = trade_data.get("tier", "Standard")

            # Create embed
            embed = DiscordEmbed(
                title=f"{self.action_emojis.get(side.lower(), '📊')} Position Opened - {tier}",
                description=f"**{symbol}** - {side.upper()}",
                color=self.tier_colors.get(tier, 0x808080),
                timestamp=datetime.utcnow(),
            )

            # v9.1 CORE: Enhanced position details
            embed.add_embed_field(
                name="Entry Price",
                value=f"${trade_data.get('entry_price', 0):.4f}",
                inline=True,
            )
            embed.add_embed_field(
                name="Position Size",
                value=f"${trade_data.get('position_size', 0):.2f}",
                inline=True,
            )
            embed.add_embed_field(
                name="Leverage", value=f"{trade_data.get('leverage', 1)}x", inline=True
            )

            # v9.1 CORE: Risk management details
            if trade_data.get("stop_loss"):
                risk_amount = abs(
                    trade_data.get("position_size", 0)
                    * (trade_data["stop_loss"] - trade_data.get("entry_price", 0))
                    / trade_data.get("entry_price", 1)
                )
                embed.add_embed_field(
                    name="Stop Loss",
                    value=f"${trade_data['stop_loss']:.4f}",
                    inline=True,
                )
                embed.add_embed_field(
                    name="Risk Amount", value=f"${risk_amount:.2f}", inline=True
                )

                # Risk percentage
                account_balance = trade_data.get("account_balance", 0)
                if account_balance > 0:
                    risk_percent = (risk_amount / account_balance) * 100
                    embed.add_embed_field(
                        name="Risk %", value=f"{risk_percent:.2f}%", inline=True
                    )

            # v9.1 CORE: Multi-TP support
            tp_fields = []
            for i in range(1, 4):
                tp_key = f"take_profit_{i}"
                if trade_data.get(tp_key):
                    tp_fields.append(f"TP{i}: ${trade_data[tp_key]:.4f}")

            if tp_fields:
                embed.add_embed_field(
                    name="Take Profits", value=" | ".join(tp_fields), inline=False
                )

            # v9.1 CORE: Signal context
            embed.add_embed_field(name="Signal Tier", value=tier, inline=True)
            if trade_data.get("signal_strength"):
                embed.add_embed_field(
                    name="Signal Strength",
                    value=f"{trade_data['signal_strength']:.3f}",
                    inline=True,
                )

            # v9.1 CORE: Order tagging
            if trade_data.get("order_tag"):
                embed.add_embed_field(
                    name="Order Tag", value=trade_data["order_tag"], inline=True
                )

            # v9.1 CORE: Enhanced market context
            if trade_data.get("market_regime"):
                regime = trade_data["market_regime"]
                regime_emoji = self.regime_emojis.get(regime, "❓")
                embed.add_embed_field(
                    name="Market Regime", value=f"{regime_emoji} {regime}", inline=True
                )

            # Send to trades channel
            result = await self._send_webhook(self.trades_webhook or self.webhook_url, embed)
    
            # Log wrapper execution
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace(
                "trade_opened", 
                result, 
                duration, 
                f"{symbol} {side} ${trade_data.get('position_size', 0):.2f}"
            )

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("trade_opened", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send trade opened notification: {e}")

    async def send_trade_closed_notification(self, trade_data: Dict) -> None:
        """Send enhanced trade closed notification - v9.1 ENHANCED"""
        try:
            symbol = trade_data.get("symbol", "UNKNOWN")
            pnl = trade_data.get("realized_pnl", 0)
            pnl_percent = trade_data.get("pnl_percent", 0)
            start_time = datetime.utcnow()
            symbol = trade_data.get("symbol", "UNKNOWN")
            pnl = trade_data.get("realized_pnl", 0)

            # v9.1 CORE: Enhanced PnL categorization
            if pnl > 0:
                color = 0x00FF00
                emoji = self.status_emojis["profit"]
                title = "Position Closed - PROFIT 💰"
            elif pnl < 0:
                color = 0xFF0000
                emoji = self.status_emojis["loss"]
                title = "Position Closed - LOSS 📉"
            else:
                color = 0x808080
                emoji = self.status_emojis["breakeven"]
                title = "Position Closed - BREAKEVEN ➖"

            # Create embed
            embed = DiscordEmbed(
                title=f"{emoji} {title}",
                description=f"**{symbol}**",
                color=color,
                timestamp=datetime.utcnow(),
            )

            # v9.1 CORE: Enhanced trade details
            embed.add_embed_field(
                name="Entry Price",
                value=f"${trade_data.get('entry_price', 0):.4f}",
                inline=True,
            )
            embed.add_embed_field(
                name="Exit Price",
                value=f"${trade_data.get('exit_price', 0):.4f}",
                inline=True,
            )
            embed.add_embed_field(
                name="Position Size",
                value=f"${trade_data.get('position_size', 0):.2f}",
                inline=True,
            )

            # v9.1 CORE: Enhanced PnL display
            embed.add_embed_field(name="Realized PnL", value=f"${pnl:.2f}", inline=True)
            embed.add_embed_field(
                name="PnL %", value=f"{pnl_percent:.2f}%", inline=True
            )

            # v9.1 CORE: Close reason with emoji
            close_reason = trade_data.get("close_reason", "Manual")
            close_emoji = {
                "TP1": "🎯",
                "TP2": "🎯",
                "TP3": "🎯",
                "SL": "🛑",
                "Manual": "👤",
                "Emergency": "🚨",
                "Timeout": "⏰",
            }.get(close_reason, "❓")

            embed.add_embed_field(
                name="Close Reason", value=f"{close_emoji} {close_reason}", inline=True
            )

            # v9.1 CORE: Trade duration
            if trade_data.get("duration_minutes"):
                duration = trade_data["duration_minutes"]
                if duration < 60:
                    duration_str = f"{duration}m"
                elif duration < 1440:
                    duration_str = f"{duration//60}h {duration%60}m"
                else:
                    duration_str = f"{duration//1440}d {(duration%1440)//60}h"

                embed.add_embed_field(name="Duration", value=duration_str, inline=True)

            # v9.1 CORE: Signal tier performance
            if trade_data.get("tier"):
                embed.add_embed_field(
                    name="Signal Tier", value=trade_data["tier"], inline=True
                )

            # v9.1 CORE: Risk-reward ratio
            if trade_data.get("risk_reward_ratio"):
                embed.add_embed_field(
                    name="R:R Ratio",
                    value=f"1:{trade_data['risk_reward_ratio']:.2f}",
                    inline=True,
                )

            # v9.1 CORE: Fees
            if trade_data.get("total_fees"):
                embed.add_embed_field(
                    name="Total Fees",
                    value=f"${trade_data['total_fees']:.2f}",
                    inline=True,
                )

            # v9.1 CORE: Performance impact
            if trade_data.get("account_balance_after"):
                balance_change = trade_data["account_balance_after"] - trade_data.get(
                    "account_balance_before", 0
                )
                embed.add_embed_field(
                    name="Balance Impact", value=f"${balance_change:.2f}", inline=True
                )

            # Send to trades channel
            result = await self._send_webhook(self.trades_webhook or self.webhook_url, embed)
    
            # Log wrapper execution
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            pnl_status = "PROFIT" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
            await self.log_notification_trace(
                f"trade_closed_{pnl_status.lower()}", 
                result, 
                duration, 
                f"{symbol} ${pnl:.2f} ({trade_data.get('close_reason', 'Manual')})"
            )


        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("trade_closed", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send trade closed notification: {e}")


    async def send_performance_report(self, stats: Dict, period: str = "daily") -> None:
        """Send enhanced performance report - v9.1 ENHANCED"""
        start_time = datetime.utcnow()
        try:
            total_trades = stats.get("total_trades", 0)
            win_rate = stats.get("win_rate", 0)
            total_pnl = stats.get("total_pnl", 0)

            # v9.1 CORE: Enhanced color logic
            if total_pnl > 100:
                color = 0x00FF00  # Green for good profit
            elif total_pnl > 0:
                color = 0x90EE90  # Light green for small profit
            elif total_pnl > -100:
                color = 0xFFB6C1  # Light red for small loss
            else:
                color = 0xFF0000  # Red for significant loss

            # Create embed
            embed = DiscordEmbed(
                title=f"📊 {period.capitalize()} Performance Report",
                description=f"Performance summary for {datetime.utcnow().strftime('%Y-%m-%d')}",
                color=color,
                timestamp=datetime.utcnow(),
            )

            # v9.1 CORE: Enhanced overall stats
            embed.add_embed_field(
                name="Total Trades", value=str(total_trades), inline=True
            )
            embed.add_embed_field(
                name="Win Rate", value=f"{win_rate:.1f}%", inline=True
            )
            embed.add_embed_field(
                name="Total PnL", value=f"${total_pnl:.2f}", inline=True
            )

            # v9.1 CORE: Enhanced detailed stats
            if stats.get("winning_trades"):
                embed.add_embed_field(
                    name="Winning Trades",
                    value=str(stats["winning_trades"]),
                    inline=True,
                )
            if stats.get("losing_trades"):
                embed.add_embed_field(
                    name="Losing Trades", value=str(stats["losing_trades"]), inline=True
                )
            if stats.get("breakeven_trades"):
                embed.add_embed_field(
                    name="Breakeven", value=str(stats["breakeven_trades"]), inline=True
                )

            # v9.1 CORE: Average trade performance
            if stats.get("avg_win"):
                embed.add_embed_field(
                    name="Avg Win", value=f"${stats['avg_win']:.2f}", inline=True
                )
            if stats.get("avg_loss"):
                embed.add_embed_field(
                    name="Avg Loss", value=f"${stats['avg_loss']:.2f}", inline=True
                )
            if stats.get("avg_trade"):
                embed.add_embed_field(
                    name="Avg Trade", value=f"${stats['avg_trade']:.2f}", inline=True
                )

            # v9.1 CORE: Risk metrics
            if stats.get("max_drawdown"):
                embed.add_embed_field(
                    name="Max Drawdown",
                    value=f"{stats['max_drawdown']:.2f}%",
                    inline=True,
                )
            if stats.get("sharpe_ratio"):
                embed.add_embed_field(
                    name="Sharpe Ratio",
                    value=f"{stats['sharpe_ratio']:.2f}",
                    inline=True,
                )
            if stats.get("profit_factor"):
                embed.add_embed_field(
                    name="Profit Factor",
                    value=f"{stats['profit_factor']:.2f}",
                    inline=True,
                )

            # v9.1 CORE: Enhanced tier breakdown
            if stats.get("tier_performance"):
                tier_lines = []
                for tier, tier_stats in stats["tier_performance"].items():
                    tier_emoji = {
                        "Emergency": "🚨",
                        "Platinum": "🥈",
                        "Premium": "🥇",
                        "Standard": "🔵",
                        "Quick": "🟢",
                    }.get(tier, "❓")

                    tier_lines.append(
                        f"{tier_emoji} **{tier}**: {tier_stats['trades']} trades, "
                        f"{tier_stats['win_rate']:.1f}% WR, ${tier_stats['pnl']:.2f}"
                    )

                if tier_lines:
                    embed.add_embed_field(
                        name="Performance by Tier",
                        value="\n".join(tier_lines),
                        inline=False,
                    )

            # v9.1 CORE: Best and worst trades
            if stats.get("best_trade"):
                embed.add_embed_field(
                    name="🏆 Best Trade",
                    value=f"${stats['best_trade']:.2f}",
                    inline=True,
                )
            if stats.get("worst_trade"):
                embed.add_embed_field(
                    name="💸 Worst Trade",
                    value=f"${stats['worst_trade']:.2f}",
                    inline=True,
                )

            # v9.1 CORE: Trading frequency
            if stats.get("avg_trades_per_day"):
                embed.add_embed_field(
                    name="Avg Trades/Day",
                    value=f"{stats['avg_trades_per_day']:.1f}",
                    inline=True,
                )

            # v9.1 CORE: Current streak
            if stats.get("current_streak"):
                streak_type = "Win" if stats["current_streak"] > 0 else "Loss"
                streak_emoji = "🔥" if stats["current_streak"] > 0 else "❄️"
                embed.add_embed_field(
                    name=f"{streak_emoji} Current Streak",
                    value=f"{abs(stats['current_streak'])} {streak_type}{'s' if abs(stats['current_streak']) > 1 else ''}",
                    inline=True,
                )

            # Send to analytics channel or main
            result = await self._send_webhook(self.analytics_webhook or self.webhook_url, embed)

            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace(
                "performance_report",
                result,
                duration,
                f"Trades={stats.get('total_trades', 0)}, PnL=${stats.get('total_pnl', 0):.2f}, WR={stats.get('win_rate', 0):.1f}%"
            )

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("performance_report", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send performance report: {e}")

    async def send_error_notification(
        self, error_message: str, title: str = "Błąd Bota", critical: bool = False
    ) -> None:
        """Send enhanced error notification - v9.1 ENHANCED"""
        start_time = datetime.utcnow()
        try:
            severity = "CRITICAL" if critical else "ERROR"
            # Create embed
            embed = DiscordEmbed(
                title=f"{self.status_emojis['error']} {'🚨 CRITICAL ERROR' if critical else 'Error'}: {title}",
                description=error_message[:4096],  # Zwiększony limit dla błędów
                color=0xFF0000 if critical else 0xFF6B6B,
                timestamp=datetime.utcnow(),
                )

            # v9.1 CORE: Enhanced context
            embed.add_embed_field(
                name="Time",
                value=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                inline=True,
            )
            embed.add_embed_field(
                name="Severity",
                value="🚨 CRITICAL" if critical else "⚠️ ERROR",
                inline=True,
            )
            embed.add_embed_field(name="Bot Version", value="v9.1", inline=True)

            # v9.1 CORE: Error categorization
            error_categories = {
                "api": "🔌 API Error",
                "database": "🗄️ Database Error",
                "network": "🌐 Network Error",
                "validation": "✅ Validation Error",
                "execution": "⚡ Execution Error",
                "system": "🖥️ System Error",
            }

            # Try to categorize error
            error_lower = error_message.lower()
            category = "system"  # default
            for cat, desc in error_categories.items():
                if cat in error_lower:
                    category = cat
                    break

            embed.add_embed_field(
                name="Category", value=error_categories[category], inline=True
            )

            # Send to errors channel or main
            result = await self._send_webhook(self.errors_webhook or self.webhook_url, embed)
    
            # Log wrapper execution
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace(
                f"error_notification_{severity.lower()}", 
                result, 
                duration, 
                f"{title}: {error_message[:100]}"
            )

        except Exception as e:
            # Użyj start_time zdefiniowanego na początku funkcji
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("error_notification", False, duration, f"Error during send_error_notification: {str(e)}")
            logger.error(f"Failed to send error notification itself: {e}")

    async def send_system_notification(
        self, message: str, notification_type: str = "info"
    ) -> None:
        """Send system notification - v9.1 ENHANCED"""
        start_time = datetime.utcnow()
        try:
            # v9.1 CORE: Enhanced type mapping
            type_config = {
                "startup": {
                    "color": 0x00FF00,
                    "emoji": "🚀",
                    "title": "System Startup",
                },
                "shutdown": {
                    "color": 0xFF0000,
                    "emoji": "🛑",
                    "title": "System Shutdown",
                },
                "pause": {"color": 0xFFFF00, "emoji": "⏸️", "title": "Bot Paused"},
                "resume": {"color": 0x00FF00, "emoji": "▶️", "title": "Bot Resumed"},
                "mode_change": {
                    "color": 0x0099FF,
                    "emoji": "⚙️",
                    "title": "Mode Changed",
                },
                "emergency": {
                    "color": 0xFF0000,
                    "emoji": "🚨",
                    "title": "Emergency Mode",
                },
                "info": {"color": 0x0099FF, "emoji": "ℹ️", "title": "Information"},
                "warning": {"color": 0xFFFF00, "emoji": "⚠️", "title": "Warning"},
                "success": {"color": 0x00FF00, "emoji": "✅", "title": "Success"},
            }

            config = type_config.get(notification_type, type_config["info"])

            # Create embed
            embed = DiscordEmbed(
                title=f"{config['emoji']} {config['title']}",
                description=message,
                color=config["color"],
                timestamp=datetime.utcnow(),
            )

            # v9.1 CORE: Add system context
            embed.add_embed_field(name="Bot Version", value="v9.1", inline=True)
            embed.add_embed_field(
                name="Timestamp",
                value=datetime.utcnow().strftime("%H:%M:%S UTC"),
                inline=True,
            )

            result = await self._send_webhook(self.webhook_url, embed)

            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace(
                f"system_{notification_type}",
                result,
                duration,
                message[:100]
            )
        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("system_notification", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send system notification: {e}")

    async def send_startup_notification(self) -> None:
        """Send enhanced bot startup notification - v9.1 ENHANCED"""
        start_time = datetime.utcnow()
        try:
            embed = DiscordEmbed(
                title="🚀 Trading Bot v9.1 Started",
                description="Enhanced trading bot is now online with v9.1 features",
                color=0x00FF00,
                timestamp=datetime.utcnow(),
            )

            # v9.1 CORE: Enhanced system info
            embed.add_embed_field(name="Version", value="9.1", inline=True)
            embed.add_embed_field(name="Indicator", value="v9.1", inline=True)
            embed.add_embed_field(name="Mode", value=Config.DEFAULT_MODE, inline=True)

            # v9.1 CORE: Enhanced configuration
            config_text = [
                f"Max Positions: {Config.MAX_POSITIONS}",
                f"Risk per Trade: {Config.RISK_PER_TRADE}%",
                f"Emergency Threshold: {Config.EMERGENCY_CLOSE_THRESHOLD}%",
                f"Tier Minimum: {Config.TIER_MINIMUM}",
            ]

            embed.add_embed_field(
                name="⚙️ Configuration", value="\n".join(config_text), inline=False
            )

            # v9.1 CORE: New features highlight
            features_text = [
                "✅ Enhanced Signal Intelligence",
                "✅ Institutional Flow Analysis",
                "✅ Fake Breakout Detection",
                "✅ Multi-Timeframe Agreement",
                "✅ Advanced Risk Management",
                "✅ Enhanced Notifications",
            ]

            embed.add_embed_field(
                name="🆕 v9.1 Features", value="\n".join(features_text), inline=False
            )

            result = await self._send_webhook(self.webhook_url, embed)

            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("startup", result, duration, "Bot started v9.1")

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("startup", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send startup notification: {e}")

    async def send_shutdown_notification(
        self, final_stats: Optional[Dict] = None
    ) -> None:
        """Send enhanced bot shutdown notification - v9.1 ENHANCED"""
        start_time = datetime.utcnow()
        try:
            embed = DiscordEmbed(
                title="🛑 Trading Bot v9.1 Shutdown",
                description="Trading bot is shutting down gracefully",
                color=0xFF0000,
                timestamp=datetime.utcnow(),
            )

            # v9.1 CORE: Enhanced shutdown stats
            if final_stats:
                embed.add_embed_field(
                    name="Session Trades",
                    value=str(final_stats.get("total_trades", 0)),
                    inline=True,
                )
                embed.add_embed_field(
                    name="Session PnL",
                    value=f"${final_stats.get('total_pnl', 0):.2f}",
                    inline=True,
                )
                embed.add_embed_field(
                    name="Win Rate",
                    value=f"{final_stats.get('win_rate', 0):.1f}%",
                    inline=True,
                )

                if final_stats.get("open_positions", 0) > 0:
                    embed.add_embed_field(
                        name="⚠️ Open Positions",
                        value=f"{final_stats['open_positions']} positions still open",
                        inline=False,
                    )

            # v9.1 CORE: Uptime
            if hasattr(self, "startup_time"):
                uptime = datetime.utcnow() - self.startup_time
                uptime_str = str(uptime).split(".")[0]  # Remove microseconds
                embed.add_embed_field(name="Uptime", value=uptime_str, inline=True)

            result = await self._send_webhook(self.webhook_url, embed)

            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("shutdown", result, duration, "Bot shutdown")

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("shutdown", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send shutdown notification: {e}")

    async def send_emergency_notification(
        self, reason: str, positions_closed: int, total_pnl: float = 0
    ) -> None:
        """Send enhanced emergency notification - v9.1 ENHANCED"""
        start_time = datetime.utcnow()
        try:
            embed = DiscordEmbed(
                title="🚨 EMERGENCY MODE ACTIVATED",
                description=f"**Reason:** {reason}",
                color=0xFF0000,
                timestamp=datetime.utcnow(),
            )

            # v9.1 CORE: Enhanced emergency details
            embed.add_embed_field(
                name="Positions Closed", value=str(positions_closed), inline=True
            )
            embed.add_embed_field(
                name="Emergency PnL", value=f"${total_pnl:.2f}", inline=True
            )
            embed.add_embed_field(
                name="Action Taken",
                value="All positions closed immediately",
                inline=True,
            )

            # v9.1 CORE: Emergency severity
            if positions_closed > 5 or abs(total_pnl) > 1000:
                embed.add_embed_field(
                    name="🚨 Severity",
                    value="HIGH - Manual review recommended",
                    inline=False,
                )

            # v9.1 CORE: Next steps
            embed.add_embed_field(
                name="📋 Next Steps",
                value="1. Review emergency logs\n2. Check system status\n3. Verify account balance\n4. Resume when safe",
                inline=False,
            )

            # Send to all channels for maximum visibility
            result = await self._send_webhook(self.webhook_url, embed)
            if self.errors_webhook:
                await self._send_webhook(self.errors_webhook, embed)
            if self.trades_webhook:
                await self._send_webhook(self.trades_webhook, embed)

            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace(
                "emergency",
                result,
                duration,
                f"Reason={reason}, Closed={positions_closed}, PnL=${total_pnl:.2f}"
            )

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("emergency", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send emergency notification: {e}")

    async def send_analytics_notification(self, analytics_data: Dict) -> None:
        """Send analytics and insights notification - v9.1 NEW"""
        try:
            start_time = datetime.utcnow()
            embed = DiscordEmbed(
                title="📈 Trading Analytics & Insights",
                description="Advanced analytics from Signal Intelligence v9.1",
                color=0x9932CC,  # Purple for analytics
                timestamp=datetime.utcnow(),
            )

            # v9.1 CORE: Signal quality metrics
            if analytics_data.get("signal_quality"):
                quality = analytics_data["signal_quality"]
                embed.add_embed_field(
                    name="📊 Signal Quality Score",
                    value=f"{quality['overall_score']:.2f}/10",
                    inline=True,
                )
                embed.add_embed_field(
                    name="🎯 Accuracy Rate",
                    value=f"{quality['accuracy_rate']:.1f}%",
                    inline=True,
                )

            # v9.1 CORE: Market regime analysis
            if analytics_data.get("regime_analysis"):
                regime = analytics_data["regime_analysis"]
                regime_emoji = self.regime_emojis.get(regime["current_regime"], "❓")
                embed.add_embed_field(
                    name="🌊 Market Regime",
                    value=f"{regime_emoji} {regime['current_regime']} ({regime['confidence']:.1%})",
                    inline=True,
                )

            # v9.1 CORE: Institutional flow insights
            if analytics_data.get("institutional_insights"):
                insights = analytics_data["institutional_insights"]
                flow_direction = "Bullish" if insights["net_flow"] > 0 else "Bearish"
                embed.add_embed_field(
                    name="🏦 Institutional Flow",
                    value=f"{flow_direction} ({abs(insights['net_flow']):.1%})",
                    inline=True,
                )

            # v9.1 CORE: Performance predictions
            if analytics_data.get("ml_predictions"):
                ml = analytics_data["ml_predictions"]
                embed.add_embed_field(
                    name="🤖 ML Win Probability",
                    value=f"{ml['win_probability']:.1%}",
                    inline=True,
                )
                embed.add_embed_field(
                    name="📈 Expected Value",
                    value=f"{ml['expected_value']:.2f}",
                    inline=True,
                )

            result = await self._send_webhook(self.analytics_webhook or self.webhook_url, embed)

            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace(
                "analytics",
                result,
                duration,
                f"Keys={list(analytics_data.keys())}"
            )

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("analytics", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send analytics notification: {e}")

    async def send_signal_decision(self, alert: dict, decision: dict) -> None:
        """
        Wrapper: używane przez main.py — mapuje na send_signal_notification
        """
        # decyzja: przyjęte jeśli udało się wykonać zlecenie
        exec_status = (decision.get("execution_result") or {}).get("status")
        accepted = exec_status == "success" or decision.get("should_trade") is True

        # powód/reason (przy odrzuceniu albo info do embed)
        reason = (
            decision.get("ml_rejection")
            or decision.get("reason")
            or (decision.get("execution_result") or {}).get("reason")
            or (decision.get("status") or "Decision processed")
        )

        trade = (decision.get("execution_result") or {}).get("order") or {}
        trade_id = trade.get("orderId")

        await self.send_signal_notification(
            alert=alert,
            accepted=accepted,
            reason=reason,
            trade_id=trade_id,
            analysis_data=decision,
        )

    async def send_entry_notification(self, order_result: dict, signal_data: dict) -> None:
        """
        Wrapper: używane przez main.py — mapuje na send_trade_opened_notification
        Buduje prosty trade_data z order_result + signal_data
        """
        trade_data = {
            "symbol": order_result.get("symbol") or signal_data.get("symbol"),
            "side": (order_result.get("side") or signal_data.get("action") or "").upper(),
            "tier": signal_data.get("tier", "Standard"),
            "entry_price": float(order_result.get("price") or signal_data.get("price") or 0),
            "position_size": float(order_result.get("origQty") or 0),
            "leverage": signal_data.get("leverage"),
            "stop_loss": signal_data.get("sl"),
            "take_profit_1": signal_data.get("tp1"),
            "take_profit_2": signal_data.get("tp2"),
            "take_profit_3": signal_data.get("tp3"),
            "signal_strength": signal_data.get("strength"),
            "order_tag": order_result.get("clientOrderId"),
            "market_regime": signal_data.get("enhanced_regime"),
        }
        await self.send_trade_opened_notification(trade_data)

    async def send_exit_notification(self, close_result: dict, reason: str = "Manual") -> None:
        """
        Prosty embed przy zamknięciu — nie mamy tu wszystkich pól,
        więc wysyłamy minimalny komunikat o zamknięciu.
        """
        try:
            from discord_webhook import DiscordEmbed
            embed = DiscordEmbed(
                title="⏹️ Position Closed",
                description=f"**{close_result.get('symbol', 'UNKNOWN')}**",
                color=0x808080
            )
            if close_result.get("price") is not None:
                embed.add_embed_field(name="Exit Price", value=f"${float(close_result['price']):.4f}", inline=True)
            embed.add_embed_field(name="Reason", value=reason, inline=True)
            await self._send_webhook(self.trades_webhook or self.webhook_url, embed)
        except Exception as e:
            logger.error(f"Failed to send exit notification: {e}")

    async def send_diagnostic_report(self, diagnostics: Dict) -> None:
        """Send comprehensive diagnostic report - v9.1 HYBRID ULTRA-DIAGNOSTICS"""
        try:
            start_time = datetime.utcnow()
            embed = DiscordEmbed(
                title="🔬 Hybrid Ultra-Diagnostics Report",
                description="Comprehensive system diagnostics and performance analysis",
                color=0x9932CC,  # Purple for diagnostics
                timestamp=datetime.utcnow(),
            )

            # System Health
            if diagnostics.get("system_health"):
                health = diagnostics["system_health"]
                health_emoji = "🟢" if health["overall_score"] > 0.8 else "🟡" if health["overall_score"] > 0.6 else "🔴"
                embed.add_embed_field(
                    name=f"{health_emoji} System Health",
                    value=f"{health['overall_score']:.1%}",
                    inline=True
                )

            # Connection Status
            if diagnostics.get("connections"):
                conn = diagnostics["connections"]
                binance_status = "🟢" if conn.get("binance") else "🔴"
                db_status = "🟢" if conn.get("database") else "🔴"
                embed.add_embed_field(
                    name="Connections",
                    value=f"Binance: {binance_status} | DB: {db_status}",
                    inline=True
                )

            # Performance Metrics
            if diagnostics.get("performance"):
                perf = diagnostics["performance"]
                embed.add_embed_field(
                    name="⚡ Performance",
                    value=f"Avg Response: {perf.get('avg_response_time', 0):.0f}ms",
                    inline=True
                )

            # ML Model Status
            if diagnostics.get("ml_status"):
                ml = diagnostics["ml_status"]
                ml_emoji = "🟢" if ml.get("model_loaded") else "🔴"
                embed.add_embed_field(
                    name=f"{ml_emoji} ML Model",
                    value=f"Accuracy: {ml.get('accuracy', 0):.1%}",
                    inline=True
                )

            # Recent Execution Traces
            if diagnostics.get("recent_traces"):
                traces = diagnostics["recent_traces"][:5]  # Last 5
                trace_lines = []
                for trace in traces:
                    status_emoji = "✅" if trace.get("success") else "❌"
                    trace_lines.append(
                        f"{status_emoji} {trace.get('operation', 'Unknown')} "
                        f"({trace.get('duration_ms', 0)}ms)"
                    )
                
                if trace_lines:
                    embed.add_embed_field(
                        name="🔍 Recent Operations",
                        value="\n".join(trace_lines),
                        inline=False
                    )

            # Pine Script Diagnostics
            if diagnostics.get("pine_diagnostics"):
                pine = diagnostics["pine_diagnostics"]
                embed.add_embed_field(
                    name="🌲 Pine Script Health",
                    value=f"Signals: {pine.get('signals_processed', 0)} | "
                          f"Errors: {pine.get('errors', 0)}",
                    inline=True
                )

            # Recommendations
            if diagnostics.get("recommendations"):
                recommendations = diagnostics["recommendations"][:3]  # Top 3
                if recommendations:
                    embed.add_embed_field(
                        name="💡 Recommendations",
                        value="\n".join([f"• {rec}" for rec in recommendations]),
                        inline=False
                    )

            result = await self._send_webhook(self.analytics_webhook or self.webhook_url, embed)

            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace(
                "diagnostic_report",
                result,
                duration,
                f"Checks={list(diagnostics.keys())}"
            )

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("diagnostic_report", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send diagnostic report: {e}")

    async def send_execution_trace_notification(self, trace_data: Dict) -> None:
        """Send execution trace notification for debugging - v9.1 DIAGNOSTICS"""
        try:
            start_time = datetime.utcnow()
            operation = trace_data.get("operation", "Unknown")
            success = trace_data.get("success", False)
            duration = trace_data.get("duration_ms", 0)
            
            # Only send notifications for important operations or failures
            if not success or operation in ["signal_processing", "trade_execution", "emergency_close"]:
                status_emoji = "✅" if success else "❌"
                color = 0x00FF00 if success else 0xFF0000
                
                embed = DiscordEmbed(
                    title=f"{status_emoji} Execution Trace: {operation}",
                    description=f"Operation completed in {duration}ms",
                    color=color,
                    timestamp=datetime.utcnow(),
                )

                if trace_data.get("details"):
                    embed.add_embed_field(
                        name="Details",
                        value=str(trace_data["details"])[:1024],
                        inline=False
                    )

                if trace_data.get("error_message"):
                    embed.add_embed_field(
                        name="Error",
                        value=trace_data["error_message"][:1024],
                        inline=False
                    )

                webhook_url = self.errors_webhook if not success else self.analytics_webhook
                result = await self._send_webhook(webhook_url or self.webhook_url, embed)

                duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                await self.log_notification_trace(
                    "execution_trace",
                    result,
                    duration,
                    f"{operation} success={success}, duration={duration}ms"
                )

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("execution_trace", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send execution trace notification: {e}")

    async def send_ml_diagnostics_notification(self, ml_data: Dict) -> None:
        """Send ML model diagnostics notification - v9.1 DIAGNOSTICS"""
        try:
            start_time = datetime.utcnow()
            embed = DiscordEmbed(
                title="🤖 ML Model Diagnostics",
                description="Machine Learning model performance and insights",
                color=0x9932CC,
                timestamp=datetime.utcnow(),
            )

            # Model Performance
            if ml_data.get("model_performance"):
                perf = ml_data["model_performance"]
                embed.add_embed_field(
                    name="📊 Model Accuracy",
                    value=f"{perf.get('accuracy', 0):.1%}",
                    inline=True
                )
                embed.add_embed_field(
                    name="🎯 Precision",
                    value=f"{perf.get('precision', 0):.1%}",
                    inline=True
                )
                embed.add_embed_field(
                    name="📈 Recall",
                    value=f"{perf.get('recall', 0):.1%}",
                    inline=True
                )

            # Feature Importance
            if ml_data.get("feature_importance"):
                features = ml_data["feature_importance"][:5]  # Top 5
                feature_lines = []
                for feature, importance in features:
                    feature_lines.append(f"• {feature}: {importance:.3f}")
                
                if feature_lines:
                    embed.add_embed_field(
                        name="🔍 Top Features",
                        value="\n".join(feature_lines),
                        inline=False
                    )

            # Recent Predictions
            if ml_data.get("recent_predictions"):
                predictions = ml_data["recent_predictions"][-3:]  # Last 3
                pred_lines = []
                for pred in predictions:
                    confidence = pred.get("confidence", 0)
                    result = "✅" if pred.get("actual_result") == pred.get("prediction") else "❌"
                    pred_lines.append(
                        f"{result} {pred.get('symbol', 'Unknown')}: "
                        f"{confidence:.1%} confidence"
                    )
                
                if pred_lines:
                    embed.add_embed_field(
                        name="🔮 Recent Predictions",
                        value="\n".join(pred_lines),
                        inline=False
                    )

            result = await self._send_webhook(self.analytics_webhook or self.webhook_url, embed)

            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace(
                "ml_diagnostics",
                result,
                duration,
                f"Metrics={list(ml_data.keys())}"
            )

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_notification_trace("ml_diagnostics", False, duration, f"Error: {str(e)}")
            logger.error(f"Failed to send ML diagnostics notification: {e}")

    async def log_notification_trace(self, operation: str, success: bool, duration_ms: int, details: str = None) -> None:
        """Log notification execution trace for diagnostics"""
        try:
            # Poprawnie wywołaj funkcję z database.py, używając argumentów `context` i `message`
            # do przekazania dodatkowych danych diagnostycznych.
            log_execution_trace(
                operation=f"discord_{operation}",
                success=success,
                message=details,
                context={
                    "duration_ms": duration_ms,
                    "source": "discord_notifier"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to log notification trace: {e}")

    async def send_performance_update(self, metrics: Dict) -> bool:
        """Wysyła aktualizację wydajności"""
        try:
            embed = DiscordEmbed(
                title="📊 Performance Update",
                color=0x3498DB,  # Blue color
                timestamp=datetime.utcnow()
            )
        
            for key, value in metrics.items():
                embed.add_embed_field(name=key.replace('_', ' ').title(), value=str(value), inline=True)
        
            return await self._send_webhook(self.alerts_webhook or self.webhook_url, embed)
        except Exception as e:
            logger.error(f"Error sending performance update: {e}")
            return False

# v9.1 CORE: Singleton instance with enhanced features
discord_notifier = DiscordNotifier()

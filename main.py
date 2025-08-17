"""
Main entry point for Binance Trading Bot v9.1
Orchestrates all components and manages the main loop
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json

import uvicorn
from concurrent.futures import ThreadPoolExecutor

from config import Config
from database import Session, init_db, cleanup_old_data, get_setting, set_setting, Trade
from binance_handler import binance_handler
from discord_notifications import DiscordNotifier
from webhook import app as webhook_app
from mode_manager import ModeManager
from analytics import AnalyticsEngine
from signal_intelligence import SignalIntelligence
from ml_predictor import TradingMLPredictor

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        """Initialize bot components"""
        self.running = False
        self.paused = False
        self.shutdown_event = asyncio.Event()
        self.discord = DiscordNotifier()
        self.mode_manager = ModeManager(self)
        self.analytics = AnalyticsEngine()
        self.signal_intelligence = SignalIntelligence()
        self.ml_predictor = TradingMLPredictor() if Config.USE_ML_FOR_DECISION else None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Runtime settings
        self.runtime_risk = Config.RISK_PER_TRADE
        self.dry_run = Config.DRY_RUN
        self.blacklisted_symbols = set()
        self.last_signal = None
        self.active_positions = {}
        
        # Task intervals (seconds)
        self.health_check_interval = 30
        self.position_check_interval = 10
        self.cleanup_interval = 3600  # 1 hour
        self.analytics_interval = 300  # 5 minutes
        
        logger.info("Trading Bot v9.1 initialized")
    
    async def start(self):
        """Start the trading bot"""
        try:
            logger.info("Starting Trading Bot v9.1...")
            
            # Initialize database
            init_db()
            
            # Validate configuration
            if not self._validate_config():
                logger.error("Configuration validation failed")
                return
            
            # Test Binance connection
            if not binance_handler.validate_api_credentials():
                logger.error("Binance API credentials validation failed")
                return
            
            # Initialize async components
            await binance_handler.init_async()
            
            # Set bot as running
            self.running = True
            with Session() as session:
                set_setting(session, "last_restart", datetime.utcnow().isoformat(), "string", "system")
            
            # Send startup notification
            await self.discord.send_startup_notification()
            
            # Start webhook server in background
            webhook_task = asyncio.create_task(self._run_webhook_server())
            
            # Start main tasks
            tasks = [
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._position_monitor_loop()),
                asyncio.create_task(self._cleanup_loop()),
                asyncio.create_task(self._analytics_loop()),
            ]
            
            logger.info("Bot started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cleanup
            logger.info("Shutting down bot...")
            self.running = False
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Close webhook server
            webhook_task.cancel()
            await asyncio.gather(webhook_task, return_exceptions=True)
            
            # Close executor
            self.executor.shutdown(wait=True)
            
            logger.info("Bot shutdown complete")
            
        except Exception as e:
            logger.error(f"Fatal error in bot start: {e}", exc_info=True)
            await self.discord.send_error_notification(f"Bot crashed: {e}")
            raise
    
    def _validate_config(self) -> bool:
        """Validate configuration"""
        errors = Config.validate()
        if errors:
            for error in errors:
                logger.error(f"Config error: {error}")
            return False
        
        logger.info("Configuration validated successfully")
        return True
    
    async def handle_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming trading signal"""
        try:
            # Store last signal
            self.last_signal = signal_data
            
            # Check if paused
            if self.paused:
                logger.info("Bot is paused, ignoring signal")
                return {"status": "ignored", "reason": "bot_paused"}
            
            # Check blacklist
            symbol = signal_data.get("symbol", "").upper()
            if symbol in self.blacklisted_symbols:
                logger.info(f"Symbol {symbol} is blacklisted, ignoring signal")
                return {"status": "ignored", "reason": "blacklisted_symbol"}
            
            # Analyze signal with SignalIntelligence
            decision = self.signal_intelligence.analyze_signal(signal_data)
            
            # Apply ML prediction if enabled
            if Config.USE_ML_FOR_DECISION and self.ml_predictor:
                ml_prediction = self.ml_predictor.predict(signal_data)
                should_take, reason = self.ml_predictor.should_take_trade(
                    ml_prediction,
                    min_win_prob=Config.ML_MIN_WIN_PROB
                )
                
                if not should_take:
                    logger.info(f"ML rejected trade: {reason}")
                    decision["should_trade"] = False
                    decision["ml_rejection"] = reason
                
                decision["ml_prediction"] = ml_prediction
            
            # Get trade parameters from ModeManager
            if decision.get("should_trade"):
                trade_params = self.mode_manager.get_trade_parameters(decision)
                decision.update(trade_params)
                
                # Execute trade
                result = await self._execute_trade(signal_data, decision)
                decision["execution_result"] = result
            
            # Send notification
            await self.discord.send_signal_decision(signal_data, decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error handling signal: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    async def _execute_trade(self, signal_data: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade based on signal and decision"""
        try:
            symbol = signal_data.get("symbol", "").upper()
            action = signal_data.get("action", "").lower()
            
            # Check if we already have a position
            if symbol in self.active_positions and Config.SINGLE_POSITION_PER_SYMBOL:
                logger.warning(f"Already have position for {symbol}, skipping")
                return {"status": "skipped", "reason": "position_exists"}
            
            # Check max concurrent positions
            if len(self.active_positions) >= Config.MAX_CONCURRENT_SLOTS:
                logger.warning(f"Max positions reached ({Config.MAX_CONCURRENT_SLOTS}), skipping")
                return {"status": "skipped", "reason": "max_positions"}
            
            # Prepare order parameters
            leverage = decision.get("leverage", Config.DEFAULT_LEVERAGE)
            risk_percent = decision.get("risk_percent", Config.RISK_PER_TRADE)
            
            # Calculate position size
            account_balance = await binance_handler.get_futures_balance()
            risk_amount = account_balance * risk_percent
            
            # Get current price
            current_price = await binance_handler.get_current_price(symbol)
            
            # Calculate stop loss and take profit levels
            sl_price = self._calculate_stop_loss(current_price, action, signal_data)
            tp_levels = self._calculate_take_profits(current_price, action, signal_data)
            
            # Calculate position size based on risk
            position_size = self._calculate_position_size(
                risk_amount, current_price, sl_price, leverage
            )
            
            # Place order
            if self.dry_run:
                logger.info(f"DRY RUN: Would place {action} order for {symbol}")
                order_result = {
                    "orderId": f"DRY_{int(time.time())}",
                    "symbol": symbol,
                    "side": action.upper(),
                    "price": current_price,
                    "origQty": position_size,
                    "status": "FILLED"
                }
            else:
                order_result = await binance_handler.place_futures_order(
                    symbol=symbol,
                    side=action.upper(),
                    quantity=position_size,
                    leverage=leverage
                )
            
            # Store position
            self.active_positions[symbol] = {
                "order_id": order_result["orderId"],
                "entry_price": current_price,
                "quantity": position_size,
                "side": action,
                "sl_price": sl_price,
                "tp_levels": tp_levels,
                "entry_time": datetime.utcnow()
            }
            
            # Save to database
            self._save_trade_to_db(signal_data, decision, order_result)
            
            # Send entry notification
            await self.discord.send_entry_notification(order_result, signal_data)
            
            return {
                "status": "success",
                "order": order_result,
                "position": self.active_positions[symbol]
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def _calculate_stop_loss(self, price: float, action: str, signal_data: Dict) -> float:
        """Calculate stop loss price"""
        sl_percent = signal_data.get("sl_percent", 2.0) / 100
        
        if action.lower() == "buy":
            return price * (1 - sl_percent)
        else:
            return price * (1 + sl_percent)
    
    def _calculate_take_profits(self, price: float, action: str, signal_data: Dict) -> List[float]:
        """Calculate take profit levels"""
        tp_rr_levels = Config.TP_RR_LEVELS if Config.USE_ALERT_LEVELS else [1.5, 3.0, 5.0]
        tp_levels = []
        
        for rr in tp_rr_levels:
            if action.lower() == "buy":
                tp_price = price * (1 + (rr * 0.02))  # 2% per RR
            else:
                tp_price = price * (1 - (rr * 0.02))
            tp_levels.append(tp_price)
        
        return tp_levels
    
    def _calculate_position_size(self, risk_amount: float, entry_price: float, 
                                 sl_price: float, leverage: int) -> float:
        """Calculate position size based on risk"""
        price_diff = abs(entry_price - sl_price)
        risk_per_unit = price_diff / entry_price
        position_value = risk_amount / risk_per_unit
        position_size = position_value / entry_price
        
        return round(position_size, 3)
    
    def _save_trade_to_db(self, signal_data: Dict, decision: Dict, order_result: Dict):
        """Save trade to database"""
        with Session() as session:
            trade = Trade(
                symbol=signal_data.get("symbol"),
                side=signal_data.get("action"),
                entry_price=float(order_result.get("price", 0)),
                quantity=float(order_result.get("origQty", 0)),
                leverage=decision.get("leverage", 1),
                stop_loss=decision.get("sl_price"),
                take_profit_1=decision.get("tp_levels", [None])[0],
                status="open",
                entry_time=datetime.utcnow(),
                raw_signal_data=json.dumps(signal_data),
                is_dry_run=self.dry_run
            )
            session.add(trade)
            session.commit()
            logger.info(f"Trade saved to DB with ID: {trade.id}")
    
    async def close_position(self, symbol: str, reason: str = "manual") -> Dict[str, Any]:
        """Close a specific position"""
        try:
            if symbol not in self.active_positions:
                return {"status": "error", "error": "Position not found"}
            
            position = self.active_positions[symbol]
            
            # Close on exchange
            if not self.dry_run:
                close_result = await binance_handler.close_futures_position(symbol)
            else:
                close_result = {
                    "symbol": symbol,
                    "status": "CLOSED",
                    "reason": reason
                }
            
            # Update database
            self.close_trade_in_db(symbol, close_result)
            
            # Remove from active positions
            del self.active_positions[symbol]
            
            # Send notification
            await self.discord.send_exit_notification(close_result, reason)
            
            return {"status": "success", "result": close_result}
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def close_trade_in_db(self, symbol: str, close_result: Dict):
        """Close trade in database"""
        with Session() as session:
            trade = session.query(Trade).filter(
                Trade.symbol == symbol,
                Trade.status == "open"
            ).order_by(Trade.entry_time.desc()).first()
            
            if trade:
                trade.status = "closed"
                trade.exit_time = datetime.utcnow()
                trade.exit_price = float(close_result.get("price", 0))
                
                # Calculate PnL
                if trade.side == "buy":
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                else:
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity
                
                trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
                
                session.commit()
                logger.info(f"Trade {trade.id} closed in DB with PnL: {trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            "running": self.running,
            "paused": self.paused,
            "mode": self.mode_manager.current_mode,
            "dry_run": self.dry_run,
            "active_positions": len(self.active_positions),
            "positions": list(self.active_positions.keys()),
            "risk_per_trade": f"{self.runtime_risk:.2%}",
            "blacklisted_symbols": list(self.blacklisted_symbols),
            "ml_enabled": Config.USE_ML_FOR_DECISION,
            "last_signal": self.last_signal.get("timestamp") if self.last_signal else None
        }
    
    def pause_trading(self):
        """Pause trading (ignore new signals)"""
        self.paused = True
        logger.info("Trading paused")
    
    def resume_trading(self):
        """Resume trading"""
        self.paused = False
        logger.info("Trading resumed")
    
    def get_last_signal(self) -> Optional[Dict[str, Any]]:
        """Get last received signal"""
        return self.last_signal
    
    def set_dry_run(self, enabled: bool):
        """Set dry run mode"""
        self.dry_run = enabled
        logger.info(f"Dry run mode: {enabled}")
    
    def set_risk(self, risk_percent: float):
        """Set risk per trade"""
        self.runtime_risk = max(0.001, min(0.1, risk_percent))  # 0.1% - 10%
        logger.info(f"Risk per trade set to: {self.runtime_risk:.2%}")
    
    def blacklist_symbol(self, symbol: str):
        """Add symbol to blacklist"""
        self.blacklisted_symbols.add(symbol.upper())
        logger.info(f"Symbol {symbol} blacklisted")
    
    def whitelist_symbol(self, symbol: str):
        """Remove symbol from blacklist"""
        self.blacklisted_symbols.discard(symbol.upper())
        logger.info(f"Symbol {symbol} whitelisted")
    
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        positions = []
        
        # Get from exchange
        if not self.dry_run:
            exchange_positions = await binance_handler.get_futures_positions()
            for pos in exchange_positions:
                if float(pos.get("positionAmt", 0)) != 0:
                    positions.append(pos)
        
        # Add local tracking
        for symbol, pos_data in self.active_positions.items():
            positions.append({
                "symbol": symbol,
                "side": pos_data["side"],
                "quantity": pos_data["quantity"],
                "entry_price": pos_data["entry_price"],
                "entry_time": pos_data["entry_time"].isoformat()
            })
        
        return positions
    
    async def _run_webhook_server(self):
        """Run webhook server"""
        try:
            # Pass bot instance to webhook app
            webhook_app.state.bot = self
            
            config = uvicorn.Config(
                app=webhook_app,
                host="0.0.0.0",
                port=Config.WEBHOOK_PORT,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"Webhook server error: {e}", exc_info=True)
    
    async def _health_check_loop(self):
        """Periodic health check"""
        while self.running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check Binance connection
                if not await binance_handler.check_connection():
                    logger.error("Binance connection lost")
                    await self.discord.send_error_notification("Binance connection lost")
                
                # Check database connection
                with Session() as session:
                    session.execute("SELECT 1")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}", exc_info=True)
    
    async def _position_monitor_loop(self):
        """Monitor open positions"""
        while self.running:
            try:
                await asyncio.sleep(self.position_check_interval)
                
                if self.paused or self.dry_run:
                    continue
                
                # Check each position
                for symbol in list(self.active_positions.keys()):
                    position = self.active_positions[symbol]
                    current_price = await binance_handler.get_current_price(symbol)
                    
                    # Check stop loss
                    if position["side"] == "buy":
                        if current_price <= position["sl_price"]:
                            await self.close_position(symbol, "stop_loss_hit")
                    else:
                        if current_price >= position["sl_price"]:
                            await self.close_position(symbol, "stop_loss_hit")
                    
                    # Check take profits
                    for i, tp_price in enumerate(position["tp_levels"]):
                        if position["side"] == "buy":
                            if current_price >= tp_price:
                                await self.close_position(symbol, f"tp{i+1}_hit")
                                break
                        else:
                            if current_price <= tp_price:
                                await self.close_position(symbol, f"tp{i+1}_hit")
                                break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position monitor error: {e}", exc_info=True)
    
    async def _cleanup_loop(self):
        """Periodic cleanup tasks"""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Cleanup old database records
                cleanup_old_data()
                
                # Train ML model if needed
                if self.ml_predictor:
                    self.ml_predictor.retrain_if_needed()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}", exc_info=True)
    
    async def _analytics_loop(self):
        """Generate analytics reports"""
        while self.running:
            try:
                await asyncio.sleep(self.analytics_interval)
                
                # Generate performance report
                report = self.analytics.generate_performance_report()
                
                # Send to Discord if significant
                if report.get("total_trades", 0) > 0:
                    await self.discord.send_performance_report(report)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analytics error: {e}", exc_info=True)


# Global bot instance
bot_instance: Optional[TradingBot] = None


async def main():
    """Main entry point"""
    global bot_instance
    
    try:
        # Create bot instance
        bot_instance = TradingBot()
        
        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            if bot_instance:
                bot_instance.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start bot
        await bot_instance.start()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())
"""
Futures User Data Stream Handler for Trading Bot v9.1
Handles real-time account updates and order events
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Optional, Callable

from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

from config import Config
from database import Session, Trade
from discord_notifications import discord_notifier

logger = logging.getLogger(__name__)


class FuturesUserStream:
    """Handles Binance Futures user data stream"""
    
    def __init__(self, client: AsyncClient):
        """Initialize user stream handler"""
        self.client = client
        self.bm: Optional[BinanceSocketManager] = None
        self.listen_key: Optional[str] = None
        self.socket = None
        self.running = False
        self.callbacks = {
            'ORDER_TRADE_UPDATE': [],
            'ACCOUNT_UPDATE': [],
            'MARGIN_CALL': [],
            'ACCOUNT_CONFIG_UPDATE': [],
            'STRATEGY_UPDATE': []
        }
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        logger.info("Futures User Stream handler initialized")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for specific event type"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"Registered callback for {event_type}")
    
    async def start(self):
        """Start user data stream"""
        try:
            self.running = True
            self.bm = BinanceSocketManager(self.client)
            
            # Get listen key
            self.listen_key = await self._get_listen_key()
            
            # Start keepalive task
            asyncio.create_task(self._keepalive_listen_key())
            
            # Start stream
            await self._connect_stream()
            
            logger.info("User data stream started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start user stream: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop user data stream"""
        self.running = False
        
        try:
            if self.socket:
                await self.socket.__aexit__(None, None, None)
            
            if self.listen_key:
                await self._close_listen_key()
            
            logger.info("User data stream stopped")
            
        except Exception as e:
            logger.error(f"Error stopping user stream: {e}")
    
    async def _get_listen_key(self) -> str:
        """Get new listen key"""
        try:
            response = await self.client.futures_stream_get_listen_key()
            return response['listenKey']
        except Exception as e:
            logger.error(f"Failed to get listen key: {e}")
            raise
    
    async def _keepalive_listen_key(self):
        """Keep listen key alive"""
        while self.running:
            try:
                await asyncio.sleep(1800)  # 30 minutes
                
                if self.listen_key and self.running:
                    await self.client.futures_stream_keepalive(self.listen_key)
                    logger.debug("Listen key keepalive sent")
                    
            except Exception as e:
                logger.error(f"Keepalive error: {e}")
                
                # Try to get new listen key
                try:
                    self.listen_key = await self._get_listen_key()
                    await self._reconnect_stream()
                except Exception as e2:
                    logger.error(f"Failed to refresh listen key: {e2}")
    
    async def _close_listen_key(self):
        """Close listen key"""
        try:
            if self.listen_key:
                await self.client.futures_stream_close(self.listen_key)
                logger.debug("Listen key closed")
        except Exception as e:
            logger.error(f"Error closing listen key: {e}")
    
    async def _connect_stream(self):
        """Connect to user data stream"""
        try:
            self.socket = self.bm.futures_user_socket()
            
            async with self.socket as stream:
                self.reconnect_attempts = 0
                logger.info("Connected to user data stream")
                
                while self.running:
                    try:
                        msg = await stream.recv()
                        
                        if msg:
                            await self._process_message(msg)
                            
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Stream error: {e}")
                        
                        if self.running:
                            await self._reconnect_stream()
                        break
                        
        except Exception as e:
            logger.error(f"Failed to connect stream: {e}")
            
            if self.running and self.reconnect_attempts < self.max_reconnect_attempts:
                await self._reconnect_stream()
    
    async def _reconnect_stream(self):
        """Reconnect to stream"""
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts > self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            await discord_notifier.send_error_notification(
                "User Stream Disconnected",
                "Maximum reconnection attempts reached. Manual intervention required."
            )
            return
        
        wait_time = min(60, 5 * self.reconnect_attempts)
        logger.info(f"Reconnecting in {wait_time} seconds... (attempt {self.reconnect_attempts})")
        
        await asyncio.sleep(wait_time)
        
        try:
            # Get new listen key
            self.listen_key = await self._get_listen_key()
            
            # Reconnect
            await self._connect_stream()
            
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            
            if self.running:
                await self._reconnect_stream()
    
    async def _process_message(self, msg: Dict):
        """Process stream message"""
        try:
            event_type = msg.get('e')
            
            if event_type == 'ORDER_TRADE_UPDATE':
                await self._handle_order_update(msg)
                
            elif event_type == 'ACCOUNT_UPDATE':
                await self._handle_account_update(msg)
                
            elif event_type == 'MARGIN_CALL':
                await self._handle_margin_call(msg)
                
            elif event_type == 'ACCOUNT_CONFIG_UPDATE':
                await self._handle_config_update(msg)
                
            elif event_type == 'STRATEGY_UPDATE':
                await self._handle_strategy_update(msg)
                
            elif event_type == 'listenKeyExpired':
                logger.warning("Listen key expired, reconnecting...")
                await self._reconnect_stream()
            
            # Call registered callbacks
            for callback in self.callbacks.get(event_type, []):
                try:
                    await callback(msg)
                except Exception as e:
                    logger.error(f"Callback error for {event_type}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.debug(f"Message: {msg}")
    
    async def _handle_order_update(self, msg: Dict):
        """Handle order trade update"""
        try:
            order = msg.get('o', {})
            
            symbol = order.get('s')
            order_id = order.get('i')
            client_order_id = order.get('c')
            side = order.get('S')
            order_type = order.get('o')
            status = order.get('X')
            execution_type = order.get('x')
            
            # Log order update
            logger.info(f"Order Update - Symbol: {symbol}, Status: {status}, Execution: {execution_type}")
            
            # Handle different order statuses
            if status == 'NEW':
                logger.info(f"New order placed: {symbol} {side}")
                
            elif status == 'FILLED':
                await self._handle_filled_order(order)
                
            elif status == 'PARTIALLY_FILLED':
                logger.info(f"Order partially filled: {symbol}")
                
            elif status == 'CANCELED':
                logger.info(f"Order canceled: {symbol}")
                
            elif status == 'REJECTED':
                logger.warning(f"Order rejected: {symbol}")
                await discord_notifier.send_error_notification(
                    f"Order Rejected: {symbol}",
                    f"Order {order_id} was rejected"
                )
                
            elif status == 'EXPIRED':
                logger.info(f"Order expired: {symbol}")
            
            # Update database if trade exists
            if client_order_id:
                await self._update_trade_status(client_order_id, status, order)
                
        except Exception as e:
            logger.error(f"Error handling order update: {e}")
    
    async def _handle_filled_order(self, order: Dict):
        """Handle filled order"""
        try:
            symbol = order.get('s')
            side = order.get('S')
            price = float(order.get('ap', 0))  # Average price
            quantity = float(order.get('z', 0))  # Cumulative filled quantity
            commission = float(order.get('n', 0))
            commission_asset = order.get('N')
            realized_pnl = float(order.get('rp', 0))
            
            logger.info(f"Order FILLED - {symbol} {side} @ {price}, Qty: {quantity}, PnL: {realized_pnl}")
            
            # Send Discord notification
            if realized_pnl != 0:
                # This is a closing order
                color = "green" if realized_pnl > 0 else "red"
                await discord_notifier.send_trade_notification(
                    symbol=symbol,
                    side="CLOSE",
                    price=price,
                    quantity=quantity,
                    realized_pnl=realized_pnl,
                    color=color
                )
            else:
                # This is an opening order
                await discord_notifier.send_trade_notification(
                    symbol=symbol,
                    side=side,
                    price=price,
                    quantity=quantity,
                    color="blue"
                )
                
        except Exception as e:
            logger.error(f"Error handling filled order: {e}")
    
    async def _handle_account_update(self, msg: Dict):
        """Handle account update"""
        try:
            event_reason = msg.get('m')  # Event reason
            
            # Update balances
            balances = msg.get('a', {}).get('B', [])
            for balance in balances:
                asset = balance.get('a')
                wallet_balance = float(balance.get('wb', 0))
                cross_wallet = float(balance.get('cw', 0))
                
                if asset == 'USDT':
                    logger.info(f"Balance Update - Wallet: {wallet_balance:.2f} USDT")
            
            # Update positions
            positions = msg.get('a', {}).get('P', [])
            for position in positions:
                symbol = position.get('s')
                amount = float(position.get('pa', 0))  # Position amount
                entry_price = float(position.get('ep', 0))
                unrealized_pnl = float(position.get('up', 0))
                
                if amount != 0:
                    logger.info(f"Position Update - {symbol}: {amount} @ {entry_price}, uPnL: {unrealized_pnl:.2f}")
                    
        except Exception as e:
            logger.error(f"Error handling account update: {e}")
    
    async def _handle_margin_call(self, msg: Dict):
        """Handle margin call"""
        try:
            logger.critical("MARGIN CALL RECEIVED!")
            
            # Send urgent Discord notification
            await discord_notifier.send_error_notification(
                "🚨 MARGIN CALL 🚨",
                "Immediate action required! Account is at risk of liquidation."
            )
            
            # Log details
            positions = msg.get('p', [])
            for pos in positions:
                symbol = pos.get('s')
                side = pos.get('ps')
                amount = pos.get('pa')
                margin_type = pos.get('mt')
                logger.critical(f"At risk: {symbol} {side} {amount} ({margin_type})")
                
        except Exception as e:
            logger.error(f"Error handling margin call: {e}")
    
    async def _handle_config_update(self, msg: Dict):
        """Handle account configuration update"""
        try:
            logger.info("Account configuration updated")
            
            # Handle leverage updates
            if 'ac' in msg:
                symbol = msg['ac'].get('s')
                leverage = msg['ac'].get('l')
                if symbol and leverage:
                    logger.info(f"Leverage updated for {symbol}: {leverage}x")
                    
        except Exception as e:
            logger.error(f"Error handling config update: {e}")
    
    async def _handle_strategy_update(self, msg: Dict):
        """Handle strategy update"""
        try:
            logger.info(f"Strategy update: {msg}")
            
            # Handle strategy-specific updates
            strategy_id = msg.get('si')
            strategy_type = msg.get('st')
            strategy_status = msg.get('ss')
            
            if strategy_status == 'NEW':
                logger.info(f"New strategy activated: {strategy_type}")
            elif strategy_status == 'CANCELED':
                logger.info(f"Strategy canceled: {strategy_type}")
                
        except Exception as e:
            logger.error(f"Error handling strategy update: {e}")
    
    async def _update_trade_status(self, client_order_id: str, status: str, order_data: Dict):
        """Update trade status in database"""
        try:
            with Session() as session:
                trade = session.query(Trade).filter_by(
                    order_id=client_order_id
                ).first()
                
                if trade:
                    # Update status
                    if status == 'FILLED':
                        trade.status = 'open' if order_data.get('S') == 'BUY' else 'closed'
                        trade.entry_price = float(order_data.get('ap', 0))
                        trade.quantity = float(order_data.get('z', 0))
                        
                        if trade.status == 'closed':
                            trade.exit_price = float(order_data.get('ap', 0))
                            trade.realized_pnl = float(order_data.get('rp', 0))
                            trade.closed_at = datetime.utcnow()
                            
                    elif status == 'CANCELED':
                        trade.status = 'canceled'
                        
                    elif status == 'REJECTED':
                        trade.status = 'rejected'
                    
                    session.commit()
                    logger.info(f"Updated trade {trade.id} status to {trade.status}")
                    
        except Exception as e:
            logger.error(f"Error updating trade status: {e}")


# Singleton instance
user_stream: Optional[FuturesUserStream] = None

async def initialize_user_stream(client: AsyncClient):
    """Initialize user stream"""
    global user_stream
    
    try:
        user_stream = FuturesUserStream(client)
        await user_stream.start()
        logger.info("User stream initialized")
        return user_stream
        
    except Exception as e:
        logger.error(f"Failed to initialize user stream: {e}")
        return None

async def stop_user_stream():
    """Stop user stream"""
    global user_stream
    
    if user_stream:
        await user_stream.stop()
        user_stream = None
"""
Bot entry point and orchestration for Trading Bot v9.1
Legacy compatibility wrapper for main.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import main, TradingBot

logger = logging.getLogger(__name__)


def run_bot():
    """Run the trading bot - legacy entry point"""
    logger.info("Starting bot via legacy entry point (bot.py)")
    
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Print deprecation notice
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║  NOTICE: bot.py is deprecated!                      ║
    ║  Please use: python main.py                         ║
    ║  Starting bot for backward compatibility...         ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    run_bot()
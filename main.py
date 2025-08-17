# main.py - Nowe centrum dowodzenia
import logging
import os
import threading
from logging.handlers import RotatingFileHandler

from gunicorn.app.base import BaseApplication

from bot import TradingBot
from config import Config
from discord_client import run_discord

# Importy z Twojego projektu
from webhook import app

os.makedirs("logs", exist_ok=True)
file_handler = RotatingFileHandler(
    "logs/bot.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)

# Konfiguracja loggingu
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")

# Globalna instancja bota
bot_instance = None


class StandaloneGunicorn(BaseApplication):
    """Uruchamia Gunicorna wewnątrz procesu Pythona."""

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        cfg = self.cfg
        for key, value in self.options.items():
            if key in cfg.settings and value is not None:
                cfg.set(key.lower(), value)

    def load(self):
        return self.application


def main():
    global bot_instance
    logger.info("Uruchamianie systemu...")

    # 1) Bot
    try:
        bot_instance = TradingBot()
        app.config["BOT_INSTANCE"] = bot_instance
        logger.info("Instancja TradingBot stworzona.")
    except Exception as e:
        logger.critical(f"Krytyczny błąd podczas inicjalizacji TradingBot: {e}", exc_info=True)
        return

    # 2) Wątek bota
    threading.Thread(target=bot_instance.run, daemon=True).start()
    logger.info("Wątek bota wystartował.")

    # 3) Discord
    threading.Thread(
        target=run_discord, args=(bot_instance, Config.DISCORD_BOT_TOKEN), daemon=True
    ).start()
    logger.info("Wątek Discord wystartował.")

    # 4) Gunicorn na 0.0.0.0:5000
    gunicorn_options = {
        "bind": "0.0.0.0:5000",
        "workers": 1,
        "timeout": 120,
        # 'worker_class': 'gthread', 'threads': 2,  # opcjonalnie
    }
    logger.info("Start Gunicorn...")
    StandaloneGunicorn(app, gunicorn_options).run()


if __name__ == "__main__":
    main()

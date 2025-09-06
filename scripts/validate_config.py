#!/usr/bin/env python3
"""
Walidacja konfiguracji dla Trading Bot v9.1
"""

import os
import sys
from dotenv import load_dotenv


def validate_config():
    """Waliduj wymagane zmienne konfiguracyjne"""
    # Sprawdź różne lokalizacje pliku .env
    env_paths = [".env", "config/.env", "./.env"]
    env_loaded = False

    for path in env_paths:
        if os.path.exists(path):
            load_dotenv(path)
            print(f"✅ Załadowano konfigurację z: {path}")
            env_loaded = True
            break

    if not env_loaded:
        print("⚠️ Nie znaleziono pliku .env, sprawdzam zmienne systemowe...")

    required_vars = [
        "BINANCE_API_KEY",
        "BINANCE_SECRET_KEY",
        "DISCORD_TOKEN",
        "DISCORD_WEBHOOK",
        "DATABASE_URL",
        "WEBHOOK_SECRET",
    ]

    missing = []
    present = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
        else:
            present.append(var)
            # Pokaż tylko pierwsze 10 znaków dla bezpieczeństwa
            masked_value = value[:10] + "..." if len(value) > 10 else value
            print(f"✅ {var}: {masked_value}")

    if missing:
        print(f"❌ Brakuje wymaganych zmiennych: {', '.join(missing)}")
        return False

    # Waliduj endpointy API
    api_url = os.getenv("BINANCE_API_URL", "https://fapi.binance.com")
    ws_url = os.getenv("BINANCE_WS_URL", "wss://fstream.binance.com")

    print(f"🌐 API URL: {api_url}")
    print(f"🌐 WebSocket URL: {ws_url}")

    if "testnet" in api_url.lower():
        print("⚠️ Uwaga: używasz endpointów TESTNET")
    else:
        print("✅ Używane są endpointy LIVE")

    print("✅ Walidacja konfiguracji zakończona pomyślnie")
    return True


if __name__ == "__main__":
    if not validate_config():
        sys.exit(1)

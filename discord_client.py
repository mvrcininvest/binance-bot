"""
Discord client v9.1: Zaktualizowany i w pełni funkcjonalny interfejs dowodzenia.
- Zachowano wszystkie oryginalne komendy.
- Naprawiono komendę /status.
- Dodano i zaimplementowano komendę /emergency.
- Ulepszona obsługa błędów i interakcji.
"""

import asyncio
import io
import json
import logging
import re
from contextlib import suppress
# Dodaj te importy po istniejących
from datetime import datetime, timedelta
from typing import Dict, Any, List
import discord
import matplotlib.pyplot as plt
import pandas as pd
from discord import File, app_commands
from discord.ext import commands
from sqlalchemy import desc
from binance_handler import binance_handler
from config import Config
from database import (
    Session,
    Trade,
    get_profile_performance,
    get_setting,
    log_command,
    set_setting,
    PineHealthLog,
    SystemHealth
)
from diagnostics import DiagnosticsEngine
from pattern_detector import pattern_detector
from decision_engine import DecisionEngine

logger = logging.getLogger("discord_client")


def normalize_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    s = re.sub(r"\.P$", "", s)
    s = s.replace("-", "").replace("_", "").replace("/", "")
    return s


class DiscordLogStreamHandler(logging.Handler):
    """Kolejkuje logi i wysyła je okresowo na kanał Discord."""

    def __init__(self, bot: "DiscordBot"):
        super().__init__()
        self.bot = bot  # <-- POPRAWKA: Używamy obiektu 'bot' przekazanego w argumencie
        self.log_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=500)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if self.bot.is_ready() and not self.bot.is_closed():
                with suppress(asyncio.QueueFull):
                    self.log_queue.put_nowait(msg)
        except Exception:
            pass

    async def send_logs_loop(self):
        await self.bot.wait_until_ready()
        log_channel_id = getattr(Config, "DISCORD_LOG_CHANNEL_ID", None) or getattr(Config, "DISCORD_CHANNEL_ID", 0)
        if not log_channel_id:
            return

        channel = self.bot.get_channel(int(log_channel_id))
        if not channel:
            logger.error(
                f"Nie znaleziono kanału logów o ID: {Config.DISCORD_LOG_CHANNEL_ID}"
            )
            return

        logger.info(
            f"Strumień logów będzie wysyłany na kanał: {getattr(channel, 'name', channel.id)}"
        )

        buffer = ""
        while not self.bot.is_closed():
            try:
                log_record = await asyncio.wait_for(self.log_queue.get(), timeout=2.0)
                line = f"{log_record}\n"
                if len(buffer) + len(line) > 1900:
                    await channel.send(f"```\n{buffer}\n```")
                    buffer = line
                else:
                    buffer += line
            except asyncio.TimeoutError:
                if buffer:
                    with suppress(Exception):
                        await channel.send(f"```\n{buffer}\n```")
                    buffer = ""
            except Exception:
                buffer = ""


class CommandCog(commands.Cog):
    """Wszystkie slash-commands w jednym Cog'u."""

    def __init__(self, bot: commands.Bot, main_bot_instance):
        super().__init__()
        self.bot = bot
        self.main_bot = main_bot_instance

    async def _reply(
        self,
        interaction: discord.Interaction,
        content: str | None = None,
        embed: discord.Embed | None = None,
        ephemeral: bool = True,
        file: File | None = None,
    ):
        try:
            kwargs = {"ephemeral": ephemeral}
            if content is not None:
                kwargs["content"] = content
            if embed is not None:
                kwargs["embed"] = embed
            if file is not None:
                kwargs["file"] = file

            if not interaction.response.is_done():
                await interaction.response.send_message(**kwargs)
            else:
                await interaction.followup.send(**kwargs)
        except Exception as e:
            logger.error(f"Błąd odpowiedzi Discord: {e}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "Wystąpił wewnętrzny błąd przy wysyłaniu odpowiedzi.",
                    ephemeral=True,
                )

    async def _log_action(
        self, interaction: discord.Interaction, command: str, details: str = ""
    ):
        try:
            await asyncio.to_thread(
                log_command, str(interaction.user), command, details
            )
        except Exception as e:
            logger.error(f"Błąd logowania akcji: {e}")

    @app_commands.command(
        name="config", description="Aktualna konfiguracja bota (z DB)."
    )
    async def config_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/config")
        await interaction.response.defer(ephemeral=True)

        def load_config():
            with Session() as session:
                return {
                    "dry_run": get_setting(session, "dry_run", str(Config.DRY_RUN)),
                    "risk_per_trade": get_setting(
                        session, "risk_per_trade", str(Config.RISK_PER_TRADE)
                    ),
                    "max_daily_loss": get_setting(session, "max_daily_loss", "-500.0"),
                    "use_ml_for_decision": get_setting(
                        session, "use_ml_for_decision", "False"
                    ),
                    "use_ml_for_sizing": get_setting(
                        session,
                        "use_ml_for_sizing",
                        str(getattr(Config, "USE_ML_FOR_SIZING", False)),
                    ),
                    "intelligent_mode_switcher_enabled": get_setting(
                        session, "intelligent_mode_switcher_enabled", "False"
                    ),
                    "leverage_override": get_setting(
                        session, "leverage_override", "True"
                    ),
                    "trailing_after_tp1": get_setting(
                        session, "trailing_after_tp1", "1"
                    ),
                    "trailing_distance_pct": get_setting(
                        session, "trailing_distance_pct", "0.005"
                    ),
                    "use_alert_levels": get_setting(
                        session,
                        "use_alert_levels",
                        "1" if Config.USE_ALERT_LEVELS else "0",
                    ),
                    "tp_rr_levels": get_setting(
                        session, "tp_rr_levels", ",".join(map(str, Config.TP_RR_LEVELS))
                    ),
                    "tp_split": get_setting(session, "tp_split", "0.5,0.3,0.2"),
                    "margin_per_trade_fraction": get_setting(
                        session,
                        "margin_per_trade_fraction",
                        "0.10",
                    ),
                    "margin_safety_buffer": get_setting(
                        session,
                        "margin_safety_buffer",
                        "0.05",
                    ),
                    "max_concurrent_slots": get_setting(
                        session,
                        "max_concurrent_slots",
                        str(Config.MAX_CONCURRENT_SLOTS),
                    ),
                    "default_margin_type": get_setting(
                        session, "default_margin_type", Config.DEFAULT_MARGIN_TYPE
                    ),
                }

        config_data = await asyncio.to_thread(load_config)

        def yesno(v: str) -> str:
            return "✅" if str(v).lower() in ("true", "1", "yes", "y") else "❌"

        embed = discord.Embed(title="⚙️ Konfiguracja Bota", color=discord.Color.purple())
        embed.add_field(
            name="Dry Run", value=yesno(config_data["dry_run"]), inline=True
        )
        embed.add_field(
            name="Ryzyko/Trade",
            value=f"{float(config_data['risk_per_trade']) * 100:.2f}%",
            inline=True,
        )
        embed.add_field(
            name="Max Daily Loss",
            value=f"${float(config_data['max_daily_loss']):.2f}",
            inline=True,
        )

        embed.add_field(
            name="ML Decision",
            value=yesno(config_data["use_ml_for_decision"]),
            inline=True,
        )
        embed.add_field(
            name="ML Sizing", value=yesno(config_data["use_ml_for_sizing"]), inline=True
        )
        embed.add_field(
            name="Auto Mode Switch",
            value=yesno(config_data["intelligent_mode_switcher_enabled"]),
            inline=True,
        )

        embed.add_field(
            name="Leverage Override",
            value=yesno(config_data["leverage_override"]),
            inline=True,
        )
        embed.add_field(
            name="USE_ALERT_LEVELS",
            value=yesno(config_data["use_alert_levels"]),
            inline=True,
        )
        embed.add_field(
            name="TP RR Levels", value=str(config_data["tp_rr_levels"]), inline=True
        )
        embed.add_field(
            name="TP Split", value=str(config_data["tp_split"]), inline=True
        )
        embed.add_field(
            name="Margin per trade",
            value=str(config_data["margin_per_trade_fraction"]),
            inline=True,
        )
        embed.add_field(
            name="Margin safety buffer",
            value=str(config_data["margin_safety_buffer"]),
            inline=True,
        )
        embed.add_field(
            name="Max slots",
            value=str(config_data["max_concurrent_slots"]),
            inline=True,
        )
        embed.add_field(
            name="Margin type",
            value=str(config_data["default_margin_type"]),
            inline=True,
        )

        await self._reply(interaction, embed=embed)

    @app_commands.command(
        name="slots", description="Pokaż/ustaw maksymalną liczbę równoległych pozycji."
    )
    @app_commands.describe(max_slots="Nowa wartość (opcjonalnie, 1..50)")
    async def slots_cmd(
        self, interaction: discord.Interaction, max_slots: int | None = None
    ):
        await self._log_action(
            interaction, "/slots", str(max_slots) if max_slots is not None else ""
        )
        await interaction.response.defer(ephemeral=True)

        def do_slots():
            with Session() as s:
                if max_slots is not None:
                    if not 1 <= max_slots <= 50:
                        return "Podaj wartość 1..50."
                    set_setting(s, "max_concurrent_slots", str(int(max_slots)))
                    return f"Ustawiono max_concurrent_slots = {int(max_slots)}"
                current = int(
                    get_setting(
                        s, "max_concurrent_slots", str(Config.MAX_CONCURRENT_SLOTS)
                    )
                    or Config.MAX_CONCURRENT_SLOTS
                )
                open_count = s.query(Trade).filter_by(status="open").count()
                return f"Sloty: {open_count}/{current} (open/max)"

        msg = await asyncio.to_thread(do_slots)
        await self._reply(interaction, msg)

    @app_commands.command(
        name="margin_buffer",
        description="Pokaż/ustaw margin_per_trade_fraction (np. 0.1 = 10%).",
    )
    @app_commands.describe(
        value="Nowa wartość (0..1). Gdy 0, używany jest 1/MARGIN_SLOTS."
    )
    async def margin_buffer_cmd(
        self, interaction: discord.Interaction, value: float | None = None
    ):
        await self._log_action(
            interaction, "/margin_buffer", str(value) if value is not None else ""
        )
        await interaction.response.defer(ephemeral=True)

        def do_mb():
            with Session() as s:
                if value is not None:
                    if not 0 <= value <= 1:
                        return "Wartość w zakresie 0..1."
                    set_setting(s, "margin_per_trade_fraction", str(value))
                    return f"Ustawiono margin_per_trade_fraction = {value:.4f}"
                current = float(
                    get_setting(
                        s,
                        "margin_per_trade_fraction",
                        str(Config.MARGIN_PER_TRADE_FRACTION),
                    )
                )
                safety = float(
                    get_setting(
                        s, "margin_safety_buffer", str(Config.MARGIN_SAFETY_BUFFER)
                    )
                )
                return f"margin_per_trade_fraction={current:.4f}, margin_safety_buffer={safety:.4f}"

        msg = await asyncio.to_thread(do_mb)
        await self._reply(interaction, msg)

    @app_commands.command(
        name="pause", description="Natychmiast pauzuje otwieranie NOWYCH pozycji."
    )
    async def pause_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/pause")
        await interaction.response.defer(ephemeral=True)
        await asyncio.to_thread(self.main_bot.pause_trading)
        await self._reply(interaction, "⏸️ Bot został zapauzowany. Nowe pozycje nie będą otwierane.")

    @app_commands.command(
        name="resume", description="Wznawia działanie bota po pauzie."
    )
    async def resume_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/resume")
        await interaction.response.defer(ephemeral=True)
        await asyncio.to_thread(self.main_bot.resume_trading)
        await self._reply(interaction, "▶️ Bot wznowił działanie. Nowe pozycje będą otwierane.")

    @app_commands.command(
        name="last_signal",
        description="Wyświetla pełne dane ostatniego odebranego sygnału.",
    )
    async def last_signal_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/last_signal")
        await interaction.response.defer(ephemeral=True)
        last_signal_data = await asyncio.to_thread(self.main_bot.get_last_signal)
        if not last_signal_data:
            await self._reply(interaction, "Brak danych ostatniego sygnału.")
            return
        embed = discord.Embed(
            title="Ostatni Odebrany Sygnał", color=discord.Color.blue()
        )
        pretty_json = json.dumps(last_signal_data, indent=2)
        embed.description = f"```json\n{pretty_json[:4000]}\n```"
        await self._reply(interaction, embed=embed)

    @app_commands.command(
        name="status", description="Pokazuje aktualny status i metryki bota."
    )
    async def status_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/status")
        await interaction.response.defer(ephemeral=True)
        try:
            status = await asyncio.to_thread(self.main_bot.get_status)
            embed = discord.Embed(
                title=f"✅ Status Bota v{status.get('version', '9.1')}",
                color=discord.Color.green(),
            )

            state = "🟢 Aktywny"
            if status.get("emergency_mode"):
                state = "🚨 Tryb Awaryjny"
            elif status.get("paused"):
                state = "⏸️ Zapauzowany"

            embed.add_field(name="Stan", value=state, inline=True)
            embed.add_field(name="Tryb Pracy", value=f"`{status.get('mode')}`", inline=True)
            embed.add_field(name="Dry Run", value="✅ Tak" if status.get('dry_run') else "❌ Nie", inline=True)

            embed.add_field(name="Otwarte Pozycje", value=str(status.get("active_positions", 0)), inline=True)
            embed.add_field(name="Dzienne PnL", value=f"${status.get('daily_pnl', 0):.2f}", inline=True)
            embed.add_field(name="Dzienne Transakcje", value=str(status.get("daily_trades", 0)), inline=True)

            embed.add_field(name="Ryzyko/Trade", value=status.get("risk_per_trade", "N/A"), inline=True)
            embed.add_field(name="ML Włączony", value="✅ Tak" if status.get('ml_enabled') else "❌ Nie", inline=True)
            embed.add_field(name="Ostatni Sygnał", value=status.get("last_signal", "Brak"), inline=True)

            perf = status.get("performance_metrics", {})
            perf_text = (
                f"Wszystkie: {perf.get('total_signals', 0)}\n"
                f"Przyjęte: {perf.get('signals_taken', 0)}\n"
                f"Odrzucone: {perf.get('signals_rejected', 0)}"
            )
            embed.add_field(name="Metryki Sygnałów", value=perf_text, inline=False)

            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error(f"/status error: {e}", exc_info=True)
            await self._reply(interaction, "❌ Błąd przy pobieraniu statusu.")

    @app_commands.command(
        name="emergency",
        description="Włącz/wyłącz akceptowanie sygnałów z trybu awaryjnego.",
    )
    @app_commands.describe(state="Wybierz 'on' aby włączyć lub 'off' aby wyłączyć.")
    @app_commands.choices(
        state=[
            app_commands.Choice(name="On", value="on"),
            app_commands.Choice(name="Off", value="off"),
        ]
    )
    async def emergency_cmd(self, interaction: discord.Interaction, state: str):
        await self._log_action(interaction, "/emergency", state)
        await interaction.response.defer(ephemeral=True)
        enabled = state.lower() == "on"
        if enabled:
            await asyncio.to_thread(self.main_bot.enable_emergency_mode)
            msg = "🚨 Tryb awaryjny został WŁĄCZONY. Nowe pozycje są blokowane."
        else:
            await asyncio.to_thread(self.main_bot.disable_emergency_mode)
            msg = "✅ Tryb awaryjny został WYŁĄCZONY. Bot działa normalnie."
        await self._reply(interaction, msg)

    @app_commands.command(name="dryrun", description="Włącz/wyłącz tryb symulacji.")
    @app_commands.describe(state="on/off — gdy pominięte, przełącznik (toggle).")
    async def dryrun_cmd(self, interaction: discord.Interaction, state: str | None = None):
        await self._log_action(interaction, "/dryrun", state or "(toggle)")
        await interaction.response.defer(ephemeral=True)
        try:
            if state is None:
                new_state = not self.main_bot.dry_run
            else:
                s = state.strip().lower()
                if s in ("on", "true", "1"):
                    new_state = True
                elif s in ("off", "false", "0"):
                    new_state = False
                else:
                    await self._reply(interaction, "Podaj 'on' lub 'off'.")
                    return

            await asyncio.to_thread(self.main_bot.set_dry_run, new_state)
            await self._reply(
                interaction,
                "🧪 Tryb Dry Run został WŁĄCZONY." if new_state else "💰 Tryb Dry Run został WYŁĄCZONY. Bot handluje na żywo.",
            )
        except Exception as e:
            logger.error(f"/dryrun error: {e}", exc_info=True)
            await self._reply(interaction, "❌ Błąd przy ustawianiu trybu Dry Run.")

    @app_commands.command(name="risk", description="Ustaw bazowe ryzyko na trade (%).")
    async def risk_cmd(self, interaction: discord.Interaction, percent: float):
        await self._log_action(interaction, "/risk", str(percent))
        await interaction.response.defer(ephemeral=True)
        try:
            risk_fraction = float(percent) / 100.0
            await asyncio.to_thread(self.main_bot.set_risk, risk_fraction)
            await self._reply(interaction, f"📊 Ryzyko na transakcję zostało ustawione na {percent:.2f}%.")
        except Exception as e:
            logger.error(f"/risk error: {e}", exc_info=True)
            await self._reply(interaction, "❌ Błąd przy ustawianiu ryzyka.")

    @app_commands.command(name="closeall", description="Zamyka wszystkie pozycje.")
    async def closeall_cmd(
        self, interaction: discord.Interaction, confirm: str | None = None
    ):
        await self._log_action(interaction, "/closeall", confirm or "")
        await interaction.response.defer(ephemeral=True)
        if (confirm or "").lower() not in ("tak", "yes", "y"):
            await self._reply(interaction, "Anulowano (podaj 'tak' aby potwierdzić).")
            return
        results = await asyncio.to_thread(binance_handler.close_all_positions)
        if not results:
            await self._reply(interaction, "Brak pozycji do zamknięcia.")
            return
        ok = [f"`{s}`" for s, v in results.items() if v]
        bad = [f"`{s}`" for s, v in results.items() if not v]
        parts = ["Zamknięcie wszystkich pozycji:"]
        if ok:
            parts.append(f"OK: {', '.join(ok)}")
        if bad:
            parts.append(f"Błąd: {', '.join(bad)}")
        await self._reply(interaction, "\n".join(parts))

    @app_commands.command(
        name="close", description="Zamyka pojedynczą pozycję rynkowo."
    )
    async def close_cmd(self, interaction: discord.Interaction, symbol: str):
        await self._log_action(interaction, "/close", symbol)
        await interaction.response.defer(ephemeral=True)
        norm = normalize_symbol(symbol)
        res = await self.main_bot.close_position(norm)
        if res.get("status") == "success":
            await self._reply(interaction, f"✅ Zamknięto pozycję {norm}.")
        else:
            await self._reply(
                interaction,
                f"❌ Nie udało się zamknąć pozycji {norm}: {res.get('error', 'unknown')}"
            )

    @app_commands.command(name="history", description="Historia transakcji.")
    @app_commands.describe(
        symbol="Symbol (opcjonalnie)", limit="Liczba transakcji (domyślnie 10)"
    )
    async def history_cmd(
        self,
        interaction: discord.Interaction,
        symbol: str | None = None,
        limit: int = 10,
    ):
        await self._log_action(
            interaction, "/history", f"symbol={symbol or 'all'}, limit={limit}"
        )
        await interaction.response.defer(ephemeral=True)

        def get_trade_history():
            with Session() as session:
                query = session.query(Trade).filter(Trade.status == "closed")
                if symbol:
                    query = query.filter(Trade.symbol == normalize_symbol(symbol))
                return (
                    query.order_by(desc(Trade.exit_time))
                    .limit(max(1, min(50, limit)))
                    .all()
                )

        trades = await asyncio.to_thread(get_trade_history)
        if not trades:
            await self._reply(
                interaction, "Brak historii transakcji dla podanych kryteriów."
            )
            return

        embed = discord.Embed(title="Historia Transakcji", color=discord.Color.blue())
        for trade in trades:
            pnl_color = "🟢" if (trade.pnl_usdt or 0) > 0 else "🔴"
            when = trade.exit_time.strftime("%d.%m %H:%M") if trade.exit_time else "N/A"
            embed.add_field(
                name=f"{pnl_color} {trade.symbol} ({trade.side.upper()})",
                value=f"PnL: ${trade.pnl_usdt or 0:.2f} ({trade.pnl_percentage or 0:+.2f}%)\nPowód: {trade.exit_reason or '?'}\nCzas: {when}",
                inline=True,
            )
        await self._reply(interaction, embed=embed)

    @app_commands.command(
        name="positions_live",
        description="Aktywne pozycje bezpośrednio z Binance (LIVE)",
    )
    async def positions_live_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/positions_live")
        await interaction.response.defer(ephemeral=True)
        try:
            positions = await asyncio.to_thread(binance_handler.check_positions)
            live = []
            for p in positions:
                try:
                    amt = float(p.get("positionAmt", 0))
                    if abs(amt) < 1e-12:
                        continue
                    live.append(p)
                except Exception:
                    continue

            if not live:
                await self._reply(
                    interaction, "📊 Brak aktywnych pozycji LIVE na Binance."
                )
                return

            embed = discord.Embed(
                title="📊 Aktywne Pozycje (LIVE)", color=discord.Color.orange()
            )
            for p in live[:10]:
                sym = p.get("symbol")
                amt = float(p.get("positionAmt", 0))
                side = "BUY" if amt > 0 else "SELL"
                entry = float(p.get("entryPrice", 0))
                pnl = float(p.get("unRealizedProfit", 0))
                embed.add_field(
                    name=f"{sym} ({side})",
                    value=(
                        f"Ilość: {abs(amt):.6f}\n"
                        f"Entry: ${entry:.6f}\n"
                        f"PnL niezreal.: ${pnl:+.2f}"
                    ),
                    inline=True,
                )
            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error("/positions_live error: %s", e, exc_info=True)
            await self._reply(interaction, "❌ Błąd pobierania pozycji LIVE.")

    # ---- NOWE KOMENDY DIAGNOSTYCZNE ----

    @app_commands.command(
        name="diagnostics", 
        description="Pełny raport diagnostyczny systemu"
    )
    @app_commands.describe(
        component="Konkretny komponent do analizy (opcjonalnie)",
        hours="Liczba godzin wstecz do analizy (domyślnie 24)"
    )
    async def diagnostics_cmd(
        self, 
        interaction: discord.Interaction, 
        component: str | None = None,
        hours: int = 24
    ):
        await self._log_action(interaction, "/diagnostics", f"component={component}, hours={hours}")
        await interaction.response.defer(ephemeral=True)

        try:
            # --- POCZĄTEK POPRAWKI ---
            # Używamy instancji diagnostic_manager z głównego bota, a nie tworzymy nowej
            if not hasattr(self.main_bot, 'diagnostic_manager'):
                await self._reply(interaction, "❌ Silnik diagnostyczny nie jest dostępny.")
                return

            report = await self.main_bot.diagnostic_manager.run_full_diagnostics()
            # --- KONIEC POPRAWKI ---

            embed = discord.Embed(
                title="🔍 Raport Diagnostyczny Systemu",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )

            health_report = report.get('overall_health', {})
            health_status = health_report.get('status', 'UNKNOWN')
            health_color = "🟢" if health_status == 'healthy' else "🔴" if health_status == 'critical' else "🟡"
            embed.add_field(
                name="Status Systemu",
                value=f"{health_color} {health_status.upper()}",
                inline=True
            )

            perf = report.get('execution_summary', {})
            if perf:
                embed.add_field(
                    name="Wydajność",
                    value=f"Checks: {perf.get('total_checks', 0)}\nCzas: {perf.get('execution_time_ms', 0)}ms",
                    inline=True
                )

            critical_issues = health_report.get('critical_issues', 0)
            warnings = health_report.get('warnings', 0)
            embed.add_field(
                name="Problemy",
                value=f"Krytyczne: {critical_issues}\nOstrzeżenia: {warnings}",
                inline=True
            )

            recommendations = report.get('recommendations', [])[:5]
            if recommendations:
                rec_text = "\n".join([f"• {rec}" for rec in recommendations])
                embed.add_field(
                    name="Rekomendacje",
                    value=rec_text[:1024],
                    inline=False
                )

            await self._reply(interaction, embed=embed)

        except Exception as e:
            logger.error(f"/diagnostics error: {e}", exc_info=True)
            await self._reply(interaction, "❌ Błąd przy generowaniu raportu diagnostycznego.")

    @app_commands.command(
    name="patterns",
    description="Analiza wzorców i anomalii w systemie"
    )
    @app_commands.describe(hours="Liczba godzin wstecz do analizy (domyślnie 24)")
    async def patterns_cmd(self, interaction: discord.Interaction, hours: int = 24):
        await self._log_action(interaction, "/patterns", f"hours={hours}")
        await interaction.response.defer(ephemeral=True)

        try:
            report = await pattern_detector.generate_pattern_report(hours)

            embed = discord.Embed(
                title="🔍 Analiza Wzorców i Anomalii",
                color=discord.Color.purple(),
                timestamp=datetime.now()
            )

            embed.add_field(
                name="Wzorce",
                value=f"Wykryte: {report.total_patterns}\nKrytyczne: {report.critical_patterns}",
                inline=True
            )

            embed.add_field(
                name="Anomalie",
                value=f"Wykryte: {report.anomalies_detected}",
                inline=True
            )

            risk_assessment = report.risk_assessment or {}
            risk_level = risk_assessment.get('overall_risk', 'UNKNOWN')
            risk_score = risk_assessment.get('risk_score', 0)
            risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(risk_level, "⚪")

            embed.add_field(
                name="Ocena Ryzyka",
                value=f"{risk_color} {risk_level}\nWynik: {risk_score:.2f}",
                inline=True
            )

            if report.recommendations:
                rec_text = "\n".join([f"• {rec}" for rec in report.recommendations[:5]])
                embed.add_field(
                    name="Rekomendacje",
                    value=rec_text[:1024],
                    inline=False
                )

            await self._reply(interaction, embed=embed)

        except Exception as e:
            logger.error(f"/patterns error: {e}", exc_info=True)
            await self._reply(interaction, "❌ Błąd przy analizie wzorców.")

    @app_commands.command(
        name="health",
        description="Szybki przegląd zdrowia systemu"
    )
    async def health_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/health")
        await interaction.response.defer(ephemeral=True)

        try:
            # --- POCZĄTEK POPRAWKI ---
            # Używamy instancji diagnostic_manager z głównego bota
            if not hasattr(self.main_bot, 'diagnostic_manager'):
                await self._reply(interaction, "❌ Silnik diagnostyczny nie jest dostępny.")
                return

            health_check = await self.main_bot.diagnostic_manager.run_quick_health_check()
            # --- KONIEC POPRAWKI ---

            overall_health = health_check.get('overall_health', 'UNKNOWN')
            status_colors = {
                'healthy': discord.Color.green(),
                'warning': discord.Color.orange(),
                'critical': discord.Color.red(),
                'unknown': discord.Color.greyple()
            }

            embed = discord.Embed(
                title="🏥 Szybki Przegląd Zdrowia Systemu",
                color=status_colors.get(overall_health.lower(), discord.Color.greyple()),
                timestamp=datetime.now()
            )

            status_emoji = {"healthy": "🟢", "warning": "🟡", "critical": "🔴", "unknown": "⚪"}
            embed.add_field(
                name="Status Ogólny",
                value=f"{status_emoji.get(overall_health.lower(), '⚪')} {overall_health.upper()}",
                inline=False
            )

            component_results = health_check.get('component_results', [])
            if component_results:
                healthy_count = sum(1 for r in component_results if r.get('status') == 'healthy')
                total_components = len(component_results)
                embed.add_field(
                    name="Komponenty",
                    value=f"Zdrowe: {healthy_count}/{total_components}",
                    inline=True
                )

            if health_check.get('critical_issues', 0) > 0 or health_check.get('warnings', 0) > 0:
                embed.add_field(
                    name="Problemy",
                    value=f"Krytyczne: {health_check.get('critical_issues', 0)}\nOstrzeżenia: {health_check.get('warnings', 0)}",
                    inline=True
                )

            await self._reply(interaction, embed=embed)

        except Exception as e:
            logger.error(f"/health error: {e}", exc_info=True)
            await self._reply(interaction, "❌ Błąd przy sprawdzaniu zdrowia systemu.")

    @app_commands.command(
        name="performance",
        description="Metryki wydajności systemu"
    )
    @app_commands.describe(hours="Liczba godzin wstecz do analizy (domyślnie 6)")
    async def performance_cmd(self, interaction: discord.Interaction, hours: int = 6):
        await self._log_action(interaction, "/performance", f"hours={hours}")
        await interaction.response.defer(ephemeral=True)
    
        try:
            # Pobierz metryki wydajności
            diagnostics_system = DiagnosticsEngine(bot=self.main_bot)
            metrics = await diagnostics_system.get_performance_metrics()
    
            embed = discord.Embed(
                title="📊 Metryki Wydajności Systemu",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )

            # Czasy odpowiedzi
            embed.add_field(
                name="Czasy Odpowiedzi",
                value=f"Średni: {metrics.get('avg_response_time', 0):.2f}ms\nMax: {metrics.get('max_response_time', 0):.2f}ms\nMin: {metrics.get('min_response_time', 0):.2f}ms",
                inline=True
            )
        
            # Throughput
            embed.add_field(
                name="Przepustowość",
                value=f"Req/min: {metrics.get('requests_per_minute', 0):.1f}\nSygnały/h: {metrics.get('signals_per_hour', 0):.1f}",
                inline=True
            )
        
            # Błędy
            error_rate = metrics.get('error_rate', 0) * 100
            embed.add_field(
                name="Wskaźnik Błędów",
                value=f"{error_rate:.2f}%\nCałkowite błędy: {metrics.get('total_errors', 0)}",
                inline=True
            )
        
            # Wykorzystanie zasobów
            if 'resource_usage' in metrics:
                resources = metrics['resource_usage']
                embed.add_field(
                    name="Zasoby",
                    value=f"CPU: {resources.get('cpu_percent', 0):.1f}%\nRAM: {resources.get('memory_percent', 0):.1f}%",
                    inline=True
                )
        
            # Uptime
            uptime_pct = metrics.get('uptime_percentage', 0)
            uptime_color = "🟢" if uptime_pct > 99 else "🟡" if uptime_pct > 95 else "🔴"
            embed.add_field(
                name="Dostępność",
                value=f"{uptime_color} {uptime_pct:.2f}%",
                inline=True
            )
        
            await self._reply(interaction, embed=embed)
        
        except Exception as e:
            logger.error(f"/performance error: {e}", exc_info=True)
            await self._reply(interaction, "❌ Błąd przy pobieraniu metryk wydajności.")

    @app_commands.command(
        name="decision_trace",
        description="Śledzenie ostatnich decyzji systemu"
    )
    @app_commands.describe(limit="Liczba ostatnich decyzji do wyświetlenia (domyślnie 10)")
    async def decision_trace_cmd(self, interaction: discord.Interaction, limit: int = 10):
        await self._log_action(interaction, "/decision_trace", f"limit={limit}")
        await interaction.response.defer(ephemeral=True)
    
        try:
            # Pobierz ślady decyzji
            if hasattr(self.main_bot, 'decision_engine'):
                traces = await self.main_bot.decision_engine.get_recent_decision_traces(
                    limit=min(limit, 10)
                )
            else:
                decision_engine = DecisionEngine()
                traces = await asyncio.to_thread(
                    decision_engine.get_recent_decision_traces,
                    limit=min(limit, 10)
                )
        
            if not traces:
                await self._reply(interaction, "Brak śladów decyzji do wyświetlenia.")
                return
        
            embed = discord.Embed(
                title="🧠 Ślady Decyzji Systemu",
                color=discord.Color.gold(),
                timestamp=datetime.now()
            )
        
            for i, trace in enumerate(traces, 1):
                decision_emoji = "✅" if trace.get('decision') == 'ACCEPT' else "❌"
                confidence = trace.get('confidence', 0) * 100
            
                # Główne czynniki decyzji
                factors = trace.get('decision_factors', {})
                main_factors = []
                for factor, weight in sorted(factors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
                    sign = "+" if weight > 0 else ""
                    main_factors.append(f"{factor}: {sign}{weight:.2f}")
            
                embed.add_field(
                    name=f"{decision_emoji} Decyzja #{i} - {trace.get('symbol', 'N/A')}",
                    value=f"Pewność: {confidence:.1f}%\nCzas: {trace.get('timestamp', 'N/A')}\nCzynniki: {', '.join(main_factors[:2])}",
                    inline=True
                )
        
            await self._reply(interaction, embed=embed)
        
        except Exception as e:
            logger.error(f"/decision_trace error: {e}", exc_info=True)
            await self._reply(interaction, "❌ Błąd przy pobieraniu śladów decyzji.")

    @app_commands.command(
        name="alerts",
        description="Aktywne alerty i ostrzeżenia systemu"
    )
    async def alerts_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/alerts")
        await interaction.response.defer(ephemeral=True)
    
        try:
            # Pobierz aktywne alerty
            diagnostics_system = DiagnosticsEngine(bot=self.main_bot)
            alerts = await asyncio.to_thread(diagnostics_system.get_active_alerts)
        
            if not alerts:
                embed = discord.Embed(
                    title="🔔 Alerty Systemu",
                    description="✅ Brak aktywnych alertów",
                    color=discord.Color.green()
                )
                await self._reply(interaction, embed=embed)
                return
        
            embed = discord.Embed(
                title="🔔 Aktywne Alerty Systemu",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
        
            # Grupuj alerty według ważności
            critical_alerts = [a for a in alerts if a.get('severity') == 'CRITICAL']
            warning_alerts = [a for a in alerts if a.get('severity') == 'WARNING']
            info_alerts = [a for a in alerts if a.get('severity') == 'INFO']
        
            # Alerty krytyczne
            if critical_alerts:
                critical_text = []
                for alert in critical_alerts[:5]:  # Pierwsze 5
                    critical_text.append(f"🔴 {alert.get('message', 'N/A')}")
                embed.add_field(
                    name="Krytyczne",
                    value="\n".join(critical_text),
                    inline=False
                )
        
            # Ostrzeżenia
            if warning_alerts:
                warning_text = []
                for alert in warning_alerts[:5]:  # Pierwsze 5
                    warning_text.append(f"🟡 {alert.get('message', 'N/A')}")
                embed.add_field(
                    name="Ostrzeżenia",
                    value="\n".join(warning_text),
                    inline=False
                )
        
            # Statystyki
            embed.add_field(
                name="Podsumowanie",
                value=f"Krytyczne: {len(critical_alerts)}\nOstrzeżenia: {len(warning_alerts)}\nInfo: {len(info_alerts)}",
                inline=True
            )
        
            await self._reply(interaction, embed=embed)
        
        except Exception as e:
            logger.error(f"/alerts error: {e}", exc_info=True)
            await self._reply(interaction, "❌ Błąd przy pobieraniu alertów.")

    @app_commands.command(name="quickdiag", description="Quick diagnostics v9.1")
    async def quickdiag_cmd(self, interaction: discord.Interaction):
        """Szybka diagnostyka w stylu v9.1"""
        await interaction.response.defer(ephemeral=True)
    
        try:
            # Użyj istniejącego diagnostics_engine
            diagnostics_system = DiagnosticsEngine(bot=self.main_bot)
            summary = diagnostics_system.get_summary()
        
            embed = discord.Embed(
                title="📊 System Diagnostics v9.1",
                color=discord.Color.green() if summary['status'] == 'HEALTHY' else discord.Color.orange()
            )
        
            embed.add_field(
                name="System Status",
                value=f"**{summary['status']}**",
                inline=False
            )
        
            embed.add_field(
                name="Market Regime",
                value=f"{summary['regime']} ({summary['confidence']:.1%} confidence)",
                inline=True
            )
        
            embed.add_field(
                name="Signal Strength",
                value=f"Buy: {summary['signals']['buy']:.1%}\nSell: {summary['signals']['sell']:.1%}",
                inline=True
            )
        
            embed.add_field(
                name="Active Zones",
                value=str(summary['zones']),
                inline=True
            )
        
            if summary['alerts']:
                embed.add_field(
                    name="⚠️ Alerts",
                    value="\n".join(f"• {alert}" for alert in summary['alerts']),
                    inline=False
                )
        
            await interaction.followup.send(embed=embed)
        
        except Exception as e:
            logger.error(f"Quick diagnostics error: {e}")
            await interaction.followup.send(f"❌ Error: {str(e)}")

    @app_commands.command(
        name="pinediag", 
        description="Pine Script diagnostics with health data"
    )
    async def pinediag_cmd(
        self, 
        interaction: discord.Interaction,
        symbol: str = None
    ):
        """Enhanced diagnostics with Pine Script data"""
        await interaction.response.defer(ephemeral=True)
    
        try:
            with Session() as session:
                # Pobierz ostatnie dane Pine health
                query = session.query(PineHealthLog).order_by(PineHealthLog.timestamp.desc())
                if symbol:
                    query = query.filter_by(symbol=symbol)
                pine_health = query.first()
            
                # Pobierz system health
                system_health = session.query(SystemHealth)\
                    .order_by(SystemHealth.timestamp.desc())\
                    .first()
            
                # Buduj raport
                embed = discord.Embed(
                    title="🌲 Pine Script Diagnostics",
                    color=discord.Color.green() if pine_health and pine_health.overall_health > 0.7 else discord.Color.orange(),
                    timestamp=datetime.now()
                )
            
                # Pine Script Health
                if pine_health:
                    embed.add_field(
                        name="Pine Script Health",
                        value=f"**Score:** {pine_health.overall_health:.2f}\n"
                            f"**Symbol:** {pine_health.symbol}\n"
                            f"**Timeframe:** {pine_health.timeframe}",
                        inline=True
                    )
                
                    embed.add_field(
                        name="Technical Indicators",
                        value=f"**ATR:** {pine_health.atr_value:.4f} (P{pine_health.atr_percentile:.0f})\n"
                              f"**ADX:** {pine_health.adx_value:.1f} ({pine_health.adx_trend_strength})\n"
                              f"**Volume:** {pine_health.volume_profile}",
                        inline=True
                    )
                
                    embed.add_field(
                        name="Market Regime",
                        value=f"**Regime:** {pine_health.regime_detected}\n"
                              f"**Confidence:** {pine_health.regime_confidence:.1%}\n"
                              f"**Stability:** {pine_health.regime_stability:.1%}",
                        inline=True
                    )
                
                    # Warnings
                    if pine_health.warnings:
                        warnings_text = "\n".join(f"⚠️ {w}" for w in pine_health.warnings[:5])
                        embed.add_field(
                            name="Warnings",
                            value=warnings_text,
                            inline=False
                        )
                else:
                    embed.add_field(
                        name="Status",
                        value="❌ No Pine Script health data available",
                        inline=False
                    )
            
                # System Health (opcjonalnie)
                if system_health:
                    embed.add_field(
                        name="System Overview",
                        value=f"**Overall:** {system_health.overall_health:.2f}\n"
                              f"**Signals (24h):** {system_health.signals_received}",
                        inline=False
                    )
            
                await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Pine diagnostics error: {e}")
            await interaction.followup.send(f"❌ Error: {str(e)}")

class DiscordBot(commands.Bot):
    """Główna klasa bota Discord, integrująca Cog z komendami."""

    def __init__(self, main_bot_instance):
        # --- POCZĄTEK POPRAWKI ---
        # Tworzymy obiekt intencji i jawnie włączamy wszystkie wymagane uprawnienia.
        intents = discord.Intents.default()
        intents.message_content = True  # Do czytania komend i wiadomości
        intents.members = True          # Do informacji o członkach serwera
        intents.presences = True        # Kluczowe dla statusu online/offline
        # --- KONIEC POPRAWKI ---
        
        super().__init__(command_prefix="!", intents=intents)
        self.main_bot = main_bot_instance
        self.log_handler = DiscordLogStreamHandler(self)
        self.bg_task: asyncio.Task | None = None

    async def setup_hook(self):
        await self.add_cog(CommandCog(self, self.main_bot))
        self.bg_task = asyncio.create_task(self.log_handler.send_logs_loop())
        try:
            if Config.DISCORD_GUILD_ID:
                guild = discord.Object(id=int(Config.DISCORD_GUILD_ID))
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                logger.info(
                    f"Zsynchronizowano {len(synced)} komend z serwerem ID: {Config.DISCORD_GUILD_ID}"
                )
            else:
                synced = await self.tree.sync()
                logger.info(f"Zsynchronizowano globalnie {len(synced)} komend.")
        except Exception as e:
            logger.error(f"Błąd synchronizacji komend: {e}", exc_info=True)

    async def on_ready(self):
        logger.info(f"Discord bot zalogowany jako {self.user}")

    async def on_app_command_error(
        self, interaction: discord.Interaction, error: Exception
    ):
        logger.error(
            f"Błąd komendy '{interaction.command.name}': {error}", exc_info=True
        )
        if not interaction.response.is_done():
            await interaction.response.send_message(
                "❌ Wystąpił nieoczekiwany błąd. Został on zarejestrowany.",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                "❌ Wystąpił nieoczekiwany błąd. Został on zarejestrowany.",
                ephemeral=True,
            )


async def run_discord(bot_instance, token: str):
    """Uruchamia klienta Discord z pełną konfiguracją."""
    if not token:
        logger.critical(
            "Brak tokena bota Discord. Nie można uruchomić klienta Discord."
        )
        return

    bot = DiscordBot(main_bot_instance=bot_instance)

    # Podpinamy handler logów do root loggera
    discord_handler = bot.log_handler
    discord_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    discord_handler.setFormatter(formatter)
    logging.getLogger().addHandler(discord_handler)

    try:
        await bot.start(token)
    except discord.errors.LoginFailure:
        logger.critical("BŁĄD: Niepoprawny token bota Discord. Sprawdź plik .env.")
    except Exception as e:
        logger.critical(f"Klient Discord zatrzymany z powodu błędu: {e}", exc_info=True)
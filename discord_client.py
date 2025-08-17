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

import discord
import matplotlib.pyplot as plt
import pandas as pd
from discord import File, app_commands
from discord.ext import commands
from sqlalchemy import desc

from config import Config
from database import (
    Session,
    Trade,
    get_profile_performance,
    get_setting,
    log_command,
    set_setting,
)

logger = logging.getLogger("discord_client")


def normalize_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    s = re.sub(r"\.P$", "", s)
    s = s.replace("-", "").replace("_", "").replace("/", "")
    return s


class DiscordLogStreamHandler(logging.Handler):
    """Kolejkuje logi i wysyła je okresowo na kanał Discord."""
    
    def __init__(self, bot_instance: "DiscordBot"):
        super().__init__()
        self.bot = bot_instance
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
        if not Config.DISCORD_LOG_CHANNEL_ID:
            return

        channel = self.bot.get_channel(int(Config.DISCORD_LOG_CHANNEL_ID))
        if not channel:
            logger.error(f"Nie znaleziono kanału logów o ID: {Config.DISCORD_LOG_CHANNEL_ID}")
            return

        logger.info(f"Strumień logów będzie wysyłany na kanał: {getattr(channel, 'name', channel.id)}")

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
                await interaction.response.send_message("Wystąpił wewnętrzny błąd przy wysyłaniu odpowiedzi.", ephemeral=True)

    async def _log_action(self, interaction: discord.Interaction, command: str, details: str = ""):
        try:
            await asyncio.to_thread(log_command, str(interaction.user), command, details)
        except Exception as e:
            logger.error(f"Błąd logowania akcji: {e}")

    # ---- Komendy z Twojego oryginalnego pliku (zachowane) ----

    @app_commands.command(name="slots", description="Pokaż/ustaw maksymalną liczbę równoległych pozycji.")
    @app_commands.describe(max_slots="Nowa wartość (opcjonalnie, 1..50)")
    async def slots_cmd(self, interaction: discord.Interaction, max_slots: int | None = None):
        await self._log_action(interaction, "/slots", str(max_slots) if max_slots is not None else "")
        await interaction.response.defer(ephemeral=True)
        
        def do_slots():
            with Session() as s:
                if max_slots is not None:
                    if not 1 <= max_slots <= 50: 
                        return "Podaj wartość 1..50."
                    set_setting(s, "max_concurrent_slots", str(int(max_slots)))
                    return f"Ustawiono max_concurrent_slots = {int(max_slots)}"
                current = int(get_setting(s, "max_concurrent_slots", str(Config.MAX_CONCURRENT_SLOTS)) or Config.MAX_CONCURRENT_SLOTS)
                open_count = s.query(Trade).filter_by(status="open").count()
                return f"Sloty: {open_count}/{current} (open/max)"
        
        msg = await asyncio.to_thread(do_slots)
        await self._reply(interaction, msg)

    @app_commands.command(name="margin_buffer", description="Pokaż/ustaw margin_per_trade_fraction (np. 0.1 = 10%).")
    @app_commands.describe(value="Nowa wartość (0..1). Gdy 0, używany jest 1/MARGIN_SLOTS.")
    async def margin_buffer_cmd(self, interaction: discord.Interaction, value: float | None = None):
        await self._log_action(interaction, "/margin_buffer", str(value) if value is not None else "")
        await interaction.response.defer(ephemeral=True)
        
        def do_mb():
            with Session() as s:
                if value is not None:
                    if not 0 <= value <= 1: 
                        return "Wartość w zakresie 0..1."
                    set_setting(s, "margin_per_trade_fraction", str(value))
                    return f"Ustawiono margin_per_trade_fraction = {value:.4f}"
                current = float(get_setting(s, "margin_per_trade_fraction", str(Config.MARGIN_PER_TRADE_FRACTION)))
                safety = float(get_setting(s, "margin_safety_buffer", str(Config.MARGIN_SAFETY_BUFFER)))
                return f"margin_per_trade_fraction={current:.4f}, margin_safety_buffer={safety:.4f}"
        
        msg = await asyncio.to_thread(do_mb)
        await self._reply(interaction, msg)

    # ---- Pauza / Resume / Last signal (zachowane) ----

    @app_commands.command(name="pause", description="Natychmiast pauzuje otwieranie NOWYCH pozycji.")
    async def pause_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/pause")
        await interaction.response.defer(ephemeral=True)
        msg = await asyncio.to_thread(self.main_bot.pause_trading)
        await self._reply(interaction, msg)

    @app_commands.command(name="resume", description="Wznawia działanie bota po pauzie.")
    async def resume_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/resume")
        await interaction.response.defer(ephemeral=True)
        msg = await asyncio.to_thread(self.main_bot.resume_trading)
        await self._reply(interaction, msg)

    @app_commands.command(name="last_signal", description="Wyświetla pełne dane ostatniego odebranego sygnału.")
    async def last_signal_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/last_signal")
        await interaction.response.defer(ephemeral=True)
        last_signal_data = await asyncio.to_thread(self.main_bot.get_last_signal)
        if "message" in last_signal_data:
            await self._reply(interaction, last_signal_data["message"])
            return
        embed = discord.Embed(title="Ostatni Odebrany Sygnał", color=discord.Color.blue())
        pretty_json = json.dumps(last_signal_data, indent=2)
        embed.description = f"```json\n{pretty_json[:4000]}\n```"
        await self._reply(interaction, embed=embed)

    # ---- ZAKTUALIZOWANE I NOWE KOMENDY ----

    @app_commands.command(name="status", description="Pokazuje aktualny status i metryki bota.")
    async def status_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/status")
        await interaction.response.defer(ephemeral=True)
        try:
            status = await asyncio.to_thread(self.main_bot.get_status)
            embed = discord.Embed(title="✅ Status Bota v9.1", color=discord.Color.blue())
            
            embed.add_field(name="Stan", value="🟢 Aktywny" if not status["is_paused"] else "⏸️ Zapauzowany", inline=True)
            embed.add_field(name="Tryb Pracy", value=f"`{status['mode']}`", inline=True)
            embed.add_field(name="Saldo (Total)", value=f"${status['balance']['total']:.2f}", inline=True)
            
            embed.add_field(name="Otwarte Pozycje", value=str(status["positions_count"]), inline=True)
            embed.add_field(name="PnL (24h)", value=f"${status['pnl_24h']:.2f}", inline=True)
            embed.add_field(name="Uptime", value=status["uptime"], inline=True)

            embed.add_field(name="Sygnały (Processed/Accepted)", value=f"{status['signals_processed']} / {status['signals_accepted']}", inline=True)
            embed.add_field(name="Acceptance Rate", value=status["signal_acceptance_rate"], inline=True)
            embed.set_footer(text=f"Bot działa od {self.main_bot.start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error(f"/status error: {e}", exc_info=True)
            await self._reply(interaction, "❌ Błąd przy pobieraniu statusu. Sprawdź logi bota.")

    @app_commands.command(name="emergency", description="Włącz/wyłącz akceptowanie sygnałów z trybu awaryjnego.")
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
        msg = await asyncio.to_thread(self.main_bot.toggle_emergency_mode, enabled)
        await self._reply(interaction, msg)

    # ---- Reszta Twoich komend (zachowana w całości) ----
    
    @app_commands.command(name="dryrun", description="Włącz/wyłącz tryb symulacji.")
    async def dryrun_cmd(self, interaction: discord.Interaction, state: str | None = None):
        await self._log_action(interaction, "/dryrun", state or "(toggle)")
        await interaction.response.defer(ephemeral=True)
        if state is None: 
            msg = await asyncio.to_thread(self.main_bot.toggle_dryrun)
        else:
            s = state.strip().lower()
            if s in ("on", "true", "1"): 
                msg = await asyncio.to_thread(self.main_bot.set_dry_run, True)
            elif s in ("off", "false", "0"): 
                msg = await asyncio.to_thread(self.main_bot.set_dry_run, False)
            else: 
                msg = "Podaj 'on' lub 'off'."
        await self._reply(interaction, msg)

    @app_commands.command(name="risk", description="Ustaw bazowe ryzyko na trade (%).")
    async def risk_cmd(self, interaction: discord.Interaction, percent: float):
        await self._log_action(interaction, "/risk", str(percent))
        await interaction.response.defer(ephemeral=True)
        msg = await asyncio.to_thread(self.main_bot.set_risk, percent)
        await self._reply(interaction, msg)

    @app_commands.command(name="closeall", description="Zamyka wszystkie pozycje.")
    async def closeall_cmd(self, interaction: discord.Interaction, confirm: str | None = None):
        await self._log_action(interaction, "/closeall", confirm or "")
        await interaction.response.defer(ephemeral=True)
        if (confirm or "").lower() not in ("tak", "yes", "y"):
            await self._reply(interaction, "Anulowano (podaj 'tak' aby potwierdzić).")
            return
        results = await asyncio.to_thread(self.main_bot.binance.close_all_positions)
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

    @app_commands.command(name="close", description="Zamyka pojedynczą pozycję rynkowo.")
    async def close_cmd(self, interaction: discord.Interaction, symbol: str):
        await self._log_action(interaction, "/close", symbol)
        await interaction.response.defer(ephemeral=True)
        norm = normalize_symbol(symbol)
        msg = await asyncio.to_thread(self.main_bot.close_position, norm)
        await self._reply(interaction, msg)

    @app_commands.command(name="history", description="Historia transakcji.")
    @app_commands.describe(symbol="Symbol (opcjonalnie)", limit="Liczba transakcji (domyślnie 10)")
    async def history_cmd(self, interaction: discord.Interaction, symbol: str | None = None, limit: int = 10):
        await self._log_action(interaction, "/history", f"symbol={symbol or 'all'}, limit={limit}")
        await interaction.response.defer(ephemeral=True)
        
        def get_trade_history():
            with Session() as session:
                query = session.query(Trade).filter(Trade.status == "closed")
                if symbol: 
                    query = query.filter(Trade.symbol == normalize_symbol(symbol))
                return query.order_by(desc(Trade.exit_time)).limit(max(1, min(50, limit))).all()
        
        trades = await asyncio.to_thread(get_trade_history)
        if not trades:
            await self._reply(interaction, "Brak historii transakcji dla podanych kryteriów.")
            return
        
        embed = discord.Embed(title="Historia Transakcji", color=discord.Color.blue())
        for trade in trades:
            pnl_color = "🟢" if (trade.pnl or 0) > 0 else "🔴"
            when = trade.exit_time.strftime("%d.%m %H:%M") if trade.exit_time else "N/A"
            embed.add_field(
                name=f"{pnl_color} {trade.symbol} ({trade.action.upper()})", 
                value=f"PnL: ${trade.pnl or 0:.2f} ({trade.pnl_percent or 0:+.2f}%)\nPowód: {trade.exit_reason or '?'}\nCzas: {when}", 
                inline=True
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
            positions = await asyncio.to_thread(self.main_bot.binance.check_positions)
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
                await self._reply(interaction, "📊 Brak aktywnych pozycji LIVE na Binance.")
                return

            embed = discord.Embed(title="📊 Aktywne Pozycje (LIVE)", color=discord.Color.orange())
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

    # ---- /config ----

    @app_commands.command(name="config", description="Aktualna konfiguracja bota (z DB).")
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
                    "use_ml_for_decision": get_setting(session, "use_ml_for_decision", "False"),
                    "use_ml_for_sizing": get_setting(
                        session,
                        "use_ml_for_sizing",
                        str(getattr(Config, "USE_ML_FOR_SIZING", False)),
                    ),
                    "intelligent_mode_switcher_enabled": get_setting(
                        session, "intelligent_mode_switcher_enabled", "False"
                    ),
                    "leverage_override": get_setting(session, "leverage_override", "True"),
                    "trailing_after_tp1": get_setting(session, "trailing_after_tp1", "1"),
                    "trailing_distance_pct": get_setting(session, "trailing_distance_pct", "0.005"),
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
                        str(Config.MARGIN_PER_TRADE_FRACTION),
                    ),
                    "margin_safety_buffer": get_setting(
                        session,
                        "margin_safety_buffer",
                        str(Config.MARGIN_SAFETY_BUFFER),
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
        embed.add_field(name="Dry Run", value=yesno(config_data["dry_run"]), inline=True)
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
        embed.add_field(name="TP RR Levels", value=str(config_data["tp_rr_levels"]), inline=True)
        embed.add_field(name="TP Split", value=str(config_data["tp_split"]), inline=True)
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


class DiscordBot(commands.Bot):
    """Główna klasa bota Discord, integrująca Cog z komendami."""

    def __init__(self, main_bot_instance):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
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
                logger.info(f"Zsynchronizowano {len(synced)} komend z serwerem ID: {Config.DISCORD_GUILD_ID}")
            else:
                synced = await self.tree.sync()
                logger.info(f"Zsynchronizowano globalnie {len(synced)} komend.")
        except Exception as e:
            logger.error(f"Błąd synchronizacji komend: {e}", exc_info=True)

    async def on_ready(self):
        logger.info(f"Discord bot zalogowany jako {self.user}")

    async def on_app_command_error(self, interaction: discord.Interaction, error: Exception):
        logger.error(f"Błąd komendy '{interaction.command.name}': {error}", exc_info=True)
        if not interaction.response.is_done():
            await interaction.response.send_message("❌ Wystąpił nieoczekiwany błąd. Został on zarejestrowany.", ephemeral=True)
        else:
            await interaction.followup.send("❌ Wystąpił nieoczekiwany błąd. Został on zarejestrowany.", ephemeral=True)


def run_discord(bot_instance, token: str):
    """Uruchamia klienta Discord z pełną konfiguracją."""
    if not token:
        logger.critical("Brak tokena bota Discord. Nie można uruchomić klienta Discord.")
        return
    
    bot = DiscordBot(main_bot_instance=bot_instance)
    
    # Podpinamy handler logów do root loggera
    discord_handler = bot.log_handler
    discord_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    discord_handler.setFormatter(formatter)
    logging.getLogger().addHandler(discord_handler)

    try:
        bot.run(token)
    except discord.errors.LoginFailure:
        logger.critical("BŁĄD: Niepoprawny token bota Discord. Sprawdź plik .env.")
    except Exception as e:
        logger.critical(f"Klient Discord zatrzymany z powodu błędu: {e}", exc_info=True)
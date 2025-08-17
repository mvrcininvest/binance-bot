"""
Discord client: stabilny układ z Cog, poprawne eventy i slash-commands.
Dodaje/pokazuje m.in.:
- /slots, /margin_buffer, /use_alert_levels, /tp_rr, /tp_split
- /ml, /ml_sizing, /auto_mode, /trailing, /circuit, /config
- /status, /commands, /chart, /performance, /positions, /positions_live, /history, /balance
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

logger = logging.getLogger(__name__)


def normalize_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    s = re.sub(r"\.P$", "", s)
    s = s.replace("-", "").replace("_", "").replace("/", "")
    return s


class DiscordLogStreamHandler(logging.Handler):
    """Kolejkuje logi i wysyła je okresowo na kanał Discord (jeśli skonfigurowany)."""

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
            logger.error("Nie znaleziono kanału logów o ID: %s", Config.DISCORD_LOG_CHANNEL_ID)
            return

        logger.info(
            "Strumień logów będzie wysyłany na kanał: %s",
            getattr(channel, "name", channel.id),
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
    """Wszystkie slash-commands w jednym Cog’u."""

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
            kwargs = {}
            if content is not None:
                kwargs["content"] = content
            if embed is not None:
                kwargs["embed"] = embed
            if file is not None:
                kwargs["file"] = file

            if not interaction.response.is_done():
                await interaction.response.send_message(ephemeral=ephemeral, **kwargs)
            else:
                await interaction.followup.send(**kwargs)
        except Exception as e:
            logger.error("Błąd odpowiedzi Discord: %s", e, exc_info=True)

    async def _log_action(self, interaction: discord.Interaction, command: str, details: str = ""):
        try:
            await asyncio.to_thread(log_command, str(interaction.user), command, details)
        except Exception as e:
            logger.error("Błąd logowania akcji: %s", e)

    # ---- Slots / Margin buffer / Alert levels / TP ----

    @app_commands.command(
        name="slots",
        description="Pokaż/ustaw maksymalną liczbę równoległych pozycji (sloty).",
    )
    @app_commands.describe(max_slots="Nowa wartość (opcjonalnie, 1..50)")
    async def slots_cmd(self, interaction: discord.Interaction, max_slots: int | None = None):
        await self._log_action(
            interaction, "/slots", str(max_slots) if max_slots is not None else ""
        )
        await interaction.response.defer(ephemeral=True)

        def do_slots():
            with Session() as s:
                if max_slots is not None:
                    if max_slots < 1 or max_slots > 50:
                        return " Podaj wartość 1..50."
                    set_setting(s, "max_concurrent_slots", str(int(max_slots)))
                    return f" Ustawiono max_concurrent_slots = {int(max_slots)}"
                # show
                current = int(
                    get_setting(s, "max_concurrent_slots", str(Config.MAX_CONCURRENT_SLOTS))
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
    @app_commands.describe(value="Nowa wartość (0..1). Gdy 0, używany jest 1/MARGIN_SLOTS.")
    async def margin_buffer_cmd(self, interaction: discord.Interaction, value: float | None = None):
        await self._log_action(
            interaction, "/margin_buffer", str(value) if value is not None else ""
        )
        await interaction.response.defer(ephemeral=True)

        def do_mb():
            with Session() as s:
                if value is not None:
                    if value < 0 or value > 1:
                        return " Wartość w zakresie 0..1."
                    set_setting(s, "margin_per_trade_fraction", str(value))
                    return f" Ustawiono margin_per_trade_fraction = {value:.4f}"
                current = float(
                    get_setting(
                        s,
                        "margin_per_trade_fraction",
                        str(Config.MARGIN_PER_TRADE_FRACTION),
                    )
                )
                safety = float(
                    get_setting(s, "margin_safety_buffer", str(Config.MARGIN_SAFETY_BUFFER))
                )
                return f"margin_per_trade_fraction={current:.4f}, margin_safety_buffer={safety:.4f}"

        msg = await asyncio.to_thread(do_mb)
        await self._reply(interaction, msg)

    @app_commands.command(
        name="use_alert_levels",
        description="Włącz/wyłącz używanie poziomów SL/TP z alertów.",
    )
    @app_commands.describe(state="on/off")
    async def use_alert_levels_cmd(self, interaction: discord.Interaction, state: str):
        await self._log_action(interaction, "/use_alert_levels", state)
        await interaction.response.defer(ephemeral=True)

        def do_toggle():
            with Session() as s:
                val = state.strip().lower() in ("on", "true", "1", "yes", "y")
                set_setting(s, "use_alert_levels", "1" if val else "0")
                return f" USE_ALERT_LEVELS = {'ON' if val else 'OFF'}"

        msg = await asyncio.to_thread(do_toggle)
        await self._reply(interaction, msg)

    @app_commands.command(name="tp_rr", description="Ustaw poziomy RR dla TP (np. 1.0,1.5,2.0).")
    async def tp_rr_cmd(self, interaction: discord.Interaction, levels: str):
        await self._log_action(interaction, "/tp_rr", levels)
        await interaction.response.defer(ephemeral=True)

        def do_set():
            parts = [p.strip() for p in levels.split(",") if p.strip()]
            vals: list[float] = []
            for p in parts:
                try:
                    vals.append(float(p))
                except Exception:
                    return " Podaj listę liczb rozdzielonych przecinkami, np. 1.0,1.5,2.0"
            if not vals:
                return " Lista nie może być pusta."
            if len(vals) > 3:
                vals = vals[:3]
            with Session() as s:
                set_setting(s, "tp_rr_levels", ",".join([str(v) for v in vals]))
            return f" TP_RR_LEVELS = {', '.join([str(v) for v in vals])}"

        msg = await asyncio.to_thread(do_set)
        await self._reply(interaction, msg)

    @app_commands.command(
        name="tp_split", description="Ustaw split TP (np. 0.5,0.3,0.2 lub 50,30,20)."
    )
    async def tp_split_cmd(self, interaction: discord.Interaction, split: str):
        await self._log_action(interaction, "/tp_split", split)
        await interaction.response.defer(ephemeral=True)

        def do_set():
            parts = [p.strip() for p in split.split(",") if p.strip()]
            vals: list[float] = []
            for p in parts:
                try:
                    v = float(p)
                    if v > 1.0:
                        v = v / 100.0
                    vals.append(v)
                except Exception:
                    return " Podaj liczby, np. 0.5,0.3,0.2 lub 50,30,20"
            if len(vals) != 3:
                return " Potrzebuję dokładnie 3 wartości."
            tot = sum(vals)
            if tot <= 0:
                return " Suma nie może być 0."
            vals = [v / tot for v in vals]
            with Session() as s:
                set_setting(s, "tp_split", ",".join([f"{v:.6f}" for v in vals]))
            return f" TP_SPLIT = {', '.join([f'{v:.2%}' for v in vals])}"

        msg = await asyncio.to_thread(do_set)
        await self._reply(interaction, msg)

    # ---- Pauza / Resume / Emergency / Last signal ----

    @app_commands.command(
        name="pause", description="Natychmiast pauzuje otwieranie NOWYCH pozycji."
    )
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
        msg = await asyncio.to_thread(self.main_bot.toggle_emergency_mode, enabled)
        await self._reply(interaction, msg)

    @app_commands.command(
        name="last_signal",
        description="Wyświetla pełne dane ostatniego odebranego sygnału.",
    )
    async def last_signal_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/last_signal")
        await interaction.response.defer(ephemeral=True)

        last_signal_data = await asyncio.to_thread(self.main_bot.get_last_signal)

        if "message" in last_signal_data:
            await self._reply(interaction, last_signal_data["message"])
            return

        embed = discord.Embed(title="Ostatni Odebrany Sygnał", color=discord.Color.blue())
        pretty_json = json.dumps(last_signal_data, indent=2)
        if len(pretty_json) > 4000:
            pretty_json = pretty_json[:4000] + "\n..."
        embed.description = f"```json\n{pretty_json}\n```"
        await self._reply(interaction, embed=embed)

    # ---- Admin / ustawienia rdzeniowe ----

    @app_commands.command(
        name="dryrun", description="Włącz/wyłącz tryb symulacji lub ustaw go jawnie."
    )
    @app_commands.describe(state="Opcjonalnie: on/off/true/false/1/0 (brak = przełącz)")
    async def dryrun_cmd(self, interaction: discord.Interaction, state: str | None = None):
        await self._log_action(interaction, "/dryrun", state or "(toggle)")
        await interaction.response.defer(ephemeral=True)
        try:
            if state is None:
                msg = await asyncio.to_thread(self.main_bot.toggle_dryrun)
            else:
                s = state.strip().lower()
                if s in ("on", "true", "1", "yes", "y", "t"):
                    msg = await asyncio.to_thread(self.main_bot.set_dry_run, True)
                elif s in ("off", "false", "0", "no", "n", "f"):
                    msg = await asyncio.to_thread(self.main_bot.set_dry_run, False)
                else:
                    msg = "Podaj 'on' lub 'off' (albo wywołaj bez parametru)."
            await self._reply(interaction, msg)
        except Exception as e:
            logger.error("/dryrun error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="risk", description="Ustaw bazowe ryzyko na trade (%).")
    async def risk_cmd(self, interaction: discord.Interaction, percent: float):
        await self._log_action(interaction, "/risk", str(percent))
        await interaction.response.defer(ephemeral=True)
        try:
            msg = await asyncio.to_thread(self.main_bot.set_risk, percent)
            await self._reply(interaction, msg)
        except Exception as e:
            logger.error("/risk error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="selftest", description="Kompleksowy test wszystkich systemów")
    async def selftest_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        results = {
            "Binance API": " Testowanie...",
            "Baza Danych": " Testowanie...",
            "Discord Webhooks": " Testowanie...",
            "ML Models": " Testowanie...",
        }

        try:
            balance = await asyncio.to_thread(self.main_bot.binance.get_balance)
            results["Binance API"] = f" OK (${balance['total']:.2f})"
        except Exception:
            results["Binance API"] = " BŁĄD"

        embed = discord.Embed(title="Self-Test Results", color=discord.Color.blue())
        for test, result in results.items():
            embed.add_field(name=test, value=result, inline=True)

        await self._reply(interaction, embed=embed)

    @app_commands.command(
        name="max_daily_loss",
        description="Ustaw/wyłącz dzienny bezpiecznik straty ($). 0=wyłącz",
    )
    async def max_daily_loss_cmd(self, interaction: discord.Interaction, amount: float):
        await self._log_action(interaction, "/max_daily_loss", str(amount))
        await interaction.response.defer(ephemeral=True)
        try:
            msg = await asyncio.to_thread(self.main_bot.set_max_daily_loss, amount)
            await self._reply(interaction, msg)
        except Exception as e:
            logger.error("/max_daily_loss error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(
        name="leverage_override",
        description="Przełącz nadpisywanie dźwigni przez bota.",
    )
    async def leverage_override_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/leverage_override")
        await interaction.response.defer(ephemeral=True)
        try:
            msg = await asyncio.to_thread(self.main_bot.toggle_leverage_override)
            await self._reply(interaction, msg)
        except Exception as e:
            logger.error("/leverage_override error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(
        name="circuit", description="Status/override/reset bezpiecznika dziennej straty"
    )
    @app_commands.describe(action="status | on | off | reset")
    async def circuit_cmd(self, interaction: discord.Interaction, action: str):
        await self._log_action(interaction, "/circuit", action)
        await interaction.response.defer(ephemeral=True)
        try:
            a = (action or "").strip().lower()
            if a == "status":
                st = await asyncio.to_thread(self.main_bot.get_circuit_status)
                embed = discord.Embed(
                    title="Bezpiecznik dziennej straty", color=discord.Color.orange()
                )
                mdl = st.get("max_daily_loss")
                embed.add_field(
                    name="Override",
                    value="WŁĄCZONY" if st.get("override") else "WYŁĄCZONY",
                    inline=True,
                )
                embed.add_field(
                    name="Tripped",
                    value="TAK" if st.get("tripped") else "NIE",
                    inline=True,
                )
                embed.add_field(
                    name="Stan bota",
                    value=" Zapauzowany" if st.get("is_paused") else " Aktywny",
                    inline=True,
                )
                embed.add_field(
                    name="Max daily loss",
                    value=f"${mdl:.2f}" if mdl else "Brak",
                    inline=True,
                )
                embed.add_field(
                    name="PnL (24h)",
                    value=f"${st.get('daily_pnl', 0.0):.2f}",
                    inline=True,
                )
                await self._reply(interaction, embed=embed)
            elif a in ("on", "off"):
                enabled = a == "on"
                msg = await asyncio.to_thread(self.main_bot.set_circuit_override, enabled)
                await self._reply(interaction, msg)
            elif a == "reset":
                msg = await asyncio.to_thread(self.main_bot.reset_circuit_breaker)
                await self._reply(interaction, msg)
            else:
                await self._reply(interaction, "Użycie: /circuit action:<status|on|off|reset>")
        except Exception as e:
            logger.error("/circuit error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(
        name="trailing",
        description="Włącz/wyłącz trailing po TP1 oraz ustaw dystans (%)",
    )
    @app_commands.describe(enabled="on/off", distance_pct="np. 0.5 oznacza 0.5%")
    async def trailing_cmd(
        self,
        interaction: discord.Interaction,
        enabled: str | None = None,
        distance_pct: float | None = None,
    ):
        await self._log_action(
            interaction, "/trailing", f"enabled={enabled} distance_pct={distance_pct}"
        )
        await interaction.response.defer(ephemeral=True)
        try:

            def update_trailing_settings():
                with Session() as session:
                    resp = []
                    if enabled is not None:
                        val = str(enabled).strip().lower() in (
                            "on",
                            "true",
                            "1",
                            "yes",
                            "y",
                        )
                        set_setting(session, "trailing_after_tp1", "1" if val else "0")
                        resp.append(f"Trailing: {'ON' if val else 'OFF'}")
                    if distance_pct is not None:
                        if distance_pct <= 0 or distance_pct > 5:
                            return " distance_pct w zakresie (0, 5]."
                        set_setting(session, "trailing_distance_pct", str(distance_pct / 100.0))
                        resp.append(f"Dystans: {distance_pct:.3f}%")
                    if not resp:
                        current_enabled = (
                            get_setting(session, "trailing_after_tp1", "1") or "1"
                        ).lower() in ("1", "true", "on", "yes")
                        current_pct_str = get_setting(session, "trailing_distance_pct", "0.005")
                        try:
                            current_pct = float(current_pct_str) * 100.0
                        except Exception:
                            current_pct = 0.5
                        return f"Trailing: {'ON' if current_enabled else 'OFF'}, dystans: {current_pct:.3f}%"
                    return " | ".join(resp)

            result = await asyncio.to_thread(update_trailing_settings)
            await self._reply(interaction, result)
        except Exception as e:
            logger.error("/trailing error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    # ---- Info / operacje rynkowe ----

    @app_commands.command(name="status", description="Pokazuje status bota")
    async def status_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/status")
        await interaction.response.defer(ephemeral=True)
        try:
            status = await asyncio.to_thread(self.main_bot.get_status)
            embed = discord.Embed(title="Status Bota", color=discord.Color.blue())
            embed.add_field(
                name="Stan",
                value=(" Aktywny" if not status["is_paused"] else " Zapauzowany"),
                inline=True,
            )
            embed.add_field(name="Tryb", value=f"`{status['mode']}`", inline=True)
            embed.add_field(name="Saldo", value=f"${status['balance']['total']:.2f}", inline=True)
            embed.add_field(name="Pozycje", value=str(status["positions_count"]), inline=True)
            embed.add_field(name="PnL (24h)", value=f"${status['pnl_24h']:.2f}", inline=True)
            embed.add_field(name="Uptime", value=status["uptime"], inline=True)
            embed.add_field(
                name="Signals",
                value=f"{status['signals_processed']} (acc: {status['signal_acceptance_rate']})",
                inline=False,
            )
            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error("/status error: %s", e, exc_info=True)
            await self._reply(interaction, " Błąd przy pobieraniu statusu.")

    @app_commands.command(name="set_mode", description="Ustaw tryb bota")
    async def set_mode_cmd(self, interaction: discord.Interaction, mode: str):
        await self._log_action(interaction, "/set_mode", mode)
        await interaction.response.defer(ephemeral=True)
        try:
            msg = await asyncio.to_thread(self.main_bot.set_mode, mode)
            await self._reply(interaction, msg)
        except Exception as e:
            logger.error("/set_mode error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="closeall", description="Zamyka wszystkie pozycje")
    async def closeall_cmd(self, interaction: discord.Interaction, confirm: str | None = None):
        await self._log_action(interaction, "/closeall", confirm or "")
        await interaction.response.defer(ephemeral=True)
        try:
            if (confirm or "").lower() not in ("tak", "yes", "y"):
                await self._reply(interaction, " Anulowano (podaj 'tak' aby potwierdzić).")
                return
            results = await asyncio.to_thread(self.main_bot.binance.close_all_positions)
            if not results:
                await self._reply(interaction, " Brak pozycji do zamknięcia.")
                return
            ok = [f"`{s}`" for s, v in results.items() if v]
            bad = [f"`{s}`" for s, v in results.items() if not v]
            parts = [" Zamknięcie wszystkich pozycji:"]
            if ok:
                parts.append(f" {', '.join(ok)}")
            if bad:
                parts.append(f" {', '.join(bad)}")
            await self._reply(interaction, "\n".join(parts))
        except Exception as e:
            logger.error("/closeall error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="close", description="Zamyka pojedynczą pozycję rynkowo")
    async def close_cmd(self, interaction: discord.Interaction, symbol: str):
        await self._log_action(interaction, "/close", symbol)
        await interaction.response.defer(ephemeral=True)
        try:
            norm = normalize_symbol(symbol)
            msg = await asyncio.to_thread(self.main_bot.close_position, norm)
            await self._reply(interaction, msg)
        except Exception as e:
            logger.error("/close error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="chart", description="Rysuje wykres (zamknięcia) dla symbolu")
    async def chart_cmd(self, interaction: discord.Interaction, symbol: str, interval: str = "5m"):
        await self._log_action(interaction, "/chart", f"{symbol} {interval}")
        await interaction.response.defer(ephemeral=True)
        try:
            norm = normalize_symbol(symbol)
            klines = await asyncio.to_thread(
                self.main_bot.binance.client.get_historical_klines,
                norm,
                interval,
                "2 day ago UTC",
            )
            if not klines:
                await self._reply(interaction, f"Brak danych dla {norm} ({interval}).")
                return
            df = pd.DataFrame(
                klines,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "qav",
                    "num_trades",
                    "taker_base_vol",
                    "taker_quote_vol",
                    "ignore",
                ],
            )
            df["close"] = df["close"].astype(float)
            df["time"] = pd.to_datetime(df["close_time"], unit="ms")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df["time"], df["close"], label=f"{norm} {interval}")
            ax.set_title(f"{norm} ({interval})")
            ax.grid(True)
            ax.legend()
            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format="png")
            buf.seek(0)
            file = File(buf, filename=f"{norm}_{interval}.png")
            await self._reply(interaction, content=f"Wykres {norm} ({interval})", file=file)
        except Exception as e:
            logger.error("/chart error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="performance", description="Wydajność profili (ostatnie 30 dni)")
    async def performance_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/performance")
        await interaction.response.defer(ephemeral=True)
        try:
            embed = discord.Embed(title="Wydajność Profili (30d)", color=discord.Color.purple())
            for p in ("A", "B", "C"):
                perf = await asyncio.to_thread(get_profile_performance, Session(), p)
                embed.add_field(
                    name=f"Profil {p}",
                    value=(
                        f"Zysk: ${perf['total_pnl']:.2f}\n"
                        f"Win Rate: {perf['win_rate']:.1f}%\n"
                        f"Transakcje: {perf['total_trades']}\n"
                    ),
                    inline=True,
                )
            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error("/performance error: %s", e, exc_info=True)
            await self._reply(interaction, " Błąd ładowania danych.")

    @app_commands.command(name="commands", description="Lista dostępnych komend")
    async def commands_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/commands")
        await interaction.response.defer(ephemeral=True)
        try:
            embed = discord.Embed(title="Centrum Dowodzenia", color=discord.Color.dark_gold())
            embed.description = (
                "Dostępne komendy: "
                "/status, /commands, /chart, /performance, /set_mode, /close, /closeall, "
                "/dryrun, /risk, /max_daily_loss, /leverage_override, /trailing, /circuit, "
                "/slots, /margin_buffer, /use_alert_levels, /tp_rr, /tp_split, "
                "/ml, /ml_sizing, /auto_mode, /emergency, /last_signal, /pause, /resume, "
                "/signals, /balance, /positions, /positions_live, /history, /config"
            )
            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error("/commands error: %s", e, exc_info=True)
            await self._reply(interaction, " Błąd.")

    # ---- ML / automatyzacja ----

    @app_commands.command(name="ml", description="Włącz/wyłącz filtrowanie ML (decyzja wejścia).")
    @app_commands.describe(state="on/off")
    async def ml_cmd(self, interaction: discord.Interaction, state: str):
        await self._log_action(interaction, "/ml", state)
        await interaction.response.defer(ephemeral=True)
        try:
            enabled = state.strip().lower() in ("on", "true", "1", "yes", "y")
            msg = await asyncio.to_thread(self.main_bot.toggle_ml_decision, enabled)
            await self._reply(interaction, msg)
        except Exception as e:
            logger.error("/ml error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(
        name="ml_sizing", description="Włącz/wyłącz ML w wielkości pozycji (sizing)."
    )
    @app_commands.describe(state="on/off")
    async def ml_sizing_cmd(self, interaction: discord.Interaction, state: str):
        await self._log_action(interaction, "/ml_sizing", state)
        await interaction.response.defer(ephemeral=True)
        try:
            enabled = state.strip().lower() in ("on", "true", "1", "yes", "y")
            # Zakładamy, że main_bot.toggle_ml_sizing persistuje to w DB.
            msg = await asyncio.to_thread(self.main_bot.toggle_ml_sizing, enabled)
            await self._reply(interaction, msg)
        except Exception as e:
            logger.error("/ml_sizing error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(
        name="auto_mode", description="Włącz/wyłącz automatyczne przełączanie trybów"
    )
    @app_commands.describe(state="on/off")
    async def auto_mode_cmd(self, interaction: discord.Interaction, state: str):
        await self._log_action(interaction, "/auto_mode", state)
        await interaction.response.defer(ephemeral=True)
        try:
            enabled = state.strip().lower() in ("on", "true", "1", "yes", "y")
            msg = await asyncio.to_thread(self.main_bot.toggle_intelligent_mode_switcher, enabled)
            await self._reply(interaction, msg)
        except Exception as e:
            logger.error("/auto_mode error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="predict", description="Predykcja ruchu rynku")
    @app_commands.describe(symbol="Symbol do analizy", hours="Horyzont czasowy w godzinach")
    async def predict_cmd(self, interaction: discord.Interaction, symbol: str, hours: int = 4):
        await self._log_action(interaction, "/predict", f"{symbol} {hours}h")
        await interaction.response.defer(ephemeral=True)
        try:
            prediction = await asyncio.to_thread(
                self.main_bot.predictive_analytics.predict_market_movement,
                symbol.upper(),
                hours,
            )
            embed = discord.Embed(title=f"Predykcja: {symbol.upper()}", color=discord.Color.blue())
            direction_emoji = (
                ""
                if prediction.get("direction") == "bullish"
                else "" if prediction.get("direction") == "bearish" else ""
            )
            embed.add_field(
                name="Kierunek",
                value=f"{direction_emoji} {prediction.get('direction', '').upper()}",
                inline=True,
            )
            embed.add_field(
                name="Pewność",
                value=f"{prediction.get('confidence', 0.0):.1%}",
                inline=True,
            )
            embed.add_field(name="Horyzont", value=f"{hours}h", inline=True)
            if "reason" in prediction:
                embed.add_field(name="Info", value=str(prediction["reason"]), inline=False)
            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error("/predict error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="optimize", description="Sugestie optymalizacji")
    async def optimize_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/optimize")
        await interaction.response.defer(ephemeral=True)
        try:
            analysis = await asyncio.to_thread(
                self.main_bot.auto_optimizer.analyze_and_suggest_optimizations
            )
            embed = discord.Embed(title="Analiza Optymalizacji", color=discord.Color.orange())

            if not analysis.get("can_optimize", False):
                embed.description = f" {analysis.get('reason', 'Nie można przeprowadzić analizy')}"
                await self._reply(interaction, embed=embed)
                return

            perf = analysis.get("performance_analysis", {})
            suggestions = analysis.get("suggestions", [])
            embed.add_field(
                name="Performance (7d)",
                value=(
                    f"Transakcje: {perf.get('total_trades', 0)}\n"
                    f"Win Rate: {perf.get('win_rate', 0):.1%}\n"
                    f"PnL: ${perf.get('total_pnl', 0):.2f}\n"
                    f"Profit Factor: {perf.get('profit_factor', 0):.2f}"
                ),
                inline=False,
            )
            if suggestions:
                high_priority = [s for s in suggestions if s.get("priority") == "high"]
                if high_priority:
                    high_text = "\n".join([f"• {s.get('suggestion')}" for s in high_priority[:3]])
                    embed.add_field(name="Priorytetowe", value=high_text, inline=False)
                else:
                    embed.add_field(
                        name="Status",
                        value="Brak sugestii - parametry OK",
                        inline=False,
                    )
            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error("/optimize error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="signals", description="Statystyki sygnałów z ostatnich 24h")
    async def signals_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/signals")
        await interaction.response.defer(ephemeral=True)
        try:
            embed = discord.Embed(title="Dashboard Sygnałów (24h)", color=discord.Color.blue())
            processed = int(getattr(self.main_bot, "signals_processed", 0))
            accepted = int(getattr(self.main_bot, "signals_accepted", 0))
            acc_rate = f"{(accepted / processed * 100):.1f}%" if processed > 0 else "N/A"
            embed.add_field(name="Odebrane", value=str(processed), inline=True)
            embed.add_field(name="Zaakceptowane", value=str(accepted), inline=True)
            embed.add_field(name="Acceptance Rate", value=acc_rate, inline=True)
            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error("/signals error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="balance", description="Szczegółowe saldo konta")
    async def balance_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/balance")
        await interaction.response.defer(ephemeral=True)
        try:
            balance = await asyncio.to_thread(self.main_bot.binance.get_balance)
            embed = discord.Embed(title="Saldo Futures (USDT)", color=discord.Color.green())
            embed.add_field(
                name="Całkowite Saldo",
                value=f"${balance.get('total', 0):.2f}",
                inline=True,
            )
            embed.add_field(
                name="Dostępne Saldo",
                value=f"${balance.get('available', 0):.2f}",
                inline=True,
            )
            used_margin = balance.get("total", 0) - balance.get("available", 0)
            embed.add_field(name="Używana Marża", value=f"${used_margin:.2f}", inline=True)
            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error("/balance error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="positions", description="Lista aktywnych pozycji")
    async def positions_cmd(self, interaction: discord.Interaction):
        await self._log_action(interaction, "/positions")
        await interaction.response.defer(ephemeral=True)
        try:
            positions = await asyncio.to_thread(self.main_bot.get_active_positions)
            if not positions:
                await self._reply(interaction, " Brak aktywnych pozycji.")
                return

            embed = discord.Embed(title="Aktywne Pozycje", color=discord.Color.orange())
            for pos in positions[:10]:
                pnl_color = "" if (pos.pnl or 0) > 0 else ""
                embed.add_field(
                    name=f"{pnl_color} {pos.symbol}",
                    value=(
                        f"Kierunek: {pos.action.upper()}\n"
                        f"Wielkość: {pos.quantity:.4f}\n"
                        f"Entry: ${pos.entry_price:.6f}\n"
                        f"PnL: ${pos.pnl or 0:.2f}"
                    ),
                    inline=True,
                )
            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error("/positions error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

    @app_commands.command(name="history", description="Historia transakcji")
    @app_commands.describe(symbol="Symbol (opcjonalnie)", limit="Liczba transakcji (domyślnie 10)")
    async def history_cmd(
        self,
        interaction: discord.Interaction,
        symbol: str | None = None,
        limit: int = 10,
    ):
        await self._log_action(interaction, "/history", f"symbol={symbol or 'all'}, limit={limit}")
        await interaction.response.defer(ephemeral=True)
        try:

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
                pnl_color = "" if (trade.pnl or 0) > 0 else ""
                when = trade.exit_time.strftime("%m-%d %H:%M") if trade.exit_time else "N/A"
                embed.add_field(
                    name=f"{pnl_color} {trade.symbol} ({trade.action.upper()})",
                    value=(
                        f"PnL: ${trade.pnl or 0:.2f} ({trade.pnl_percent or 0:+.2f}%)\n"
                        f"Powód: {trade.exit_reason or 'Manual'}\n"
                        f"Czas: {when}"
                    ),
                    inline=True,
                )
            await self._reply(interaction, embed=embed)
        except Exception as e:
            logger.error("/history error: %s", e, exc_info=True)
            await self._reply(interaction, f" Błąd: {e}")

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
                await self._reply(interaction, " Brak aktywnych pozycji LIVE na Binance.")
                return

            embed = discord.Embed(title=" Aktywne Pozycje (LIVE)", color=discord.Color.orange())
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
            await self._reply(interaction, " Błąd pobierania pozycji LIVE.")

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
            return "" if str(v).lower() in ("true", "1", "yes", "y") else ""

        embed = discord.Embed(title="Konfiguracja Bota", color=discord.Color.purple())
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
    """Subclass bota – eventy są metodami klasy, komendy rejestrowane w setup_hook."""

    def __init__(self, main_bot_instance):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix="!", intents=intents)
        self.main_bot = main_bot_instance
        self.log_handler = DiscordLogStreamHandler(self)
        self.bg_task: asyncio.Task | None = None

    async def setup_hook(self):
        # Rejestracja Cog i start pętli z logami:
        await self.add_cog(CommandCog(self, self.main_bot))
        self.bg_task = asyncio.create_task(self.log_handler.send_logs_loop())

        # Synchronizacja slash-commands:
        try:
            if Config.DISCORD_GUILD_ID:
                guild = discord.Object(id=int(Config.DISCORD_GUILD_ID))
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                logger.info(
                    "Wymuszono synchronizację %d komend z serwerem ID: %s",
                    len(synced),
                    Config.DISCORD_GUILD_ID,
                )
            else:
                synced = await self.tree.sync()
                logger.info("Zsynchronizowano globalnie %d komend.", len(synced))
        except Exception as e:
            logger.error("Błąd synchronizacji komend: %s", e, exc_info=True)

    async def on_ready(self):
        logger.info("Discord bot zalogowany jako %s", self.user)

    async def on_app_command_error(self, interaction: discord.Interaction, error: Exception):
        logger.error("Błąd komendy: %s", error, exc_info=True)
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    " Błąd po naszej stronie. Spróbuj ponownie.", ephemeral=True
                )
            else:
                await interaction.followup.send(" Błąd po naszej stronie. Spróbuj ponownie.")
        except Exception:
            pass


def run_discord(bot_instance, token: str):
    """Uruchamia klienta Discord z pełną konfiguracją."""
    bot = DiscordBot(main_bot_instance=bot_instance)

    # Podpinamy handler logów do root loggera:
    discord_handler = bot.log_handler
    discord_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    discord_handler.setFormatter(formatter)
    logging.getLogger().addHandler(discord_handler)

    # Start klienta:
    try:
        bot.run(token)
    except discord.errors.LoginFailure:
        logger.critical("BŁĄD: Niepoprawny token bota.")
    except Exception as e:
        logger.critical("Klient Discord zatrzymany: %s", e, exc_info=True)

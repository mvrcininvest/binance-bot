# discord_notifications.py - Stable notification system (ASCII safe)
import logging
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)


class DiscordNotificationSystem:
    def __init__(
        self,
        webhook_url: str,
        logs_webhook: str = None,
        trade_entries_webhook: str = None,
        trade_exits_webhook: str = None,
        signal_decisions_webhook: str = None,
        alerts_webhook: str = None,
        performance_webhook: str = None,
    ):
        self.webhook_url = webhook_url
        self.logs_webhook = logs_webhook or webhook_url
        self.trade_entries_webhook = trade_entries_webhook or webhook_url
        self.trade_exits_webhook = trade_exits_webhook or webhook_url
        self.signal_decisions_webhook = signal_decisions_webhook or webhook_url
        self.alerts_webhook = alerts_webhook or webhook_url
        self.performance_webhook = performance_webhook or webhook_url

        self.colors = {
            "success": 0x00FF00,
            "warning": 0xFFA500,
            "error": 0xFF0000,
            "info": 0x3498DB,
            "buy": 0x2ECC71,
            "sell": 0xE74C3C,
            "profit": 0x27AE60,
            "loss": 0xC0392B,
            "neutral": 0x95A5A6,
        }

    def _post(self, url: str, payload: dict[str, Any]):
        try:
            requests.post(url, json=payload, timeout=8)
        except Exception as e:
            logger.error(f"Discord POST error: {e}")

    def send_log(self, title: str, details: dict[str, Any] | None = None, level: str = "info"):
        color = self.colors.get(level, self.colors["info"])
        fields: list[dict[str, Any]] = []
        if details:
            fields = [{"name": str(k), "value": str(v), "inline": True} for k, v in details.items()]
        embed = {
            "title": f"[LOG] {title}",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": fields,
        }
        self._post(self.logs_webhook, {"embeds": [embed]})

    def send_signal_decision(self, decision: str, data: dict[str, Any]):
        try:
            color = self.colors["success"] if decision == "accepted" else self.colors["warning"]
            title = "Signal accepted" if decision == "accepted" else "Signal rejected"
            embed = {
                "title": f"{title}: {data.get('symbol')}",
                "color": color,
                "fields": [
                    {
                        "name": "Action",
                        "value": str(data.get("action", "?")),
                        "inline": True,
                    },
                    {
                        "name": "Strength",
                        "value": f"{float(data.get('strength', 0)):.1%}",
                        "inline": True,
                    },
                    {
                        "name": "Tier",
                        "value": str(data.get("tier", "?")),
                        "inline": True,
                    },
                    {
                        "name": "Reason",
                        "value": str(data.get("reason", "-")),
                        "inline": False,
                    },
                ],
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._post(self.signal_decisions_webhook, {"embeds": [embed]})
        except Exception as e:
            logger.error(f"Error sending signal decision: {e}")

    def send_trade_entry_notification(self, trade_data: dict[str, Any]):
        try:
            symbol = trade_data.get("symbol")
            side = trade_data.get("side")
            entry_price = float(trade_data.get("entry_price") or 0)
            size = float(trade_data.get("size") or 0)
            leverage = trade_data.get("leverage")
            rp = trade_data.get("risk_percent") or 0
            risk_fraction = float(rp) / 100.0 if float(rp) > 1.0 else float(rp)
            signal_strength = float(trade_data.get("signal_strength") or 0)
            sl_price = trade_data.get("sl_price")
            tp_details = trade_data.get("tp_details") or []

            position_value = size * entry_price
            risk_amount = position_value * risk_fraction if risk_fraction else 0.0

            embed = {
                "title": f"NEW POSITION: {symbol}",
                "color": self.colors["buy"] if side == "BUY" else self.colors["sell"],
                "timestamp": datetime.utcnow().isoformat(),
                "fields": [
                    {
                        "name": "Position",
                        "value": (
                            f"Side: {side}\n"
                            f"Size: {size:.6f}\n"
                            f"Entry: ${entry_price:.6f}\n"
                            f"Notional: ${position_value:.2f}"
                        ),
                        "inline": True,
                    },
                    {
                        "name": "Params",
                        "value": (
                            f"Leverage: {leverage}x\n"
                            f"Risk: {risk_fraction:.2%} (${risk_amount:.2f})\n"
                            f"Signal strength: {signal_strength:.2f}/1.0"
                        ),
                        "inline": True,
                    },
                ],
            }

            if sl_price:
                sl_price = float(sl_price)
                sl_dist_pct = (
                    abs((sl_price - entry_price) / entry_price * 100) if entry_price else 0.0
                )
                embed["fields"].append(
                    {
                        "name": "Stop Loss",
                        "value": f"SL: ${sl_price:.6f}\nDistance: {sl_dist_pct:.2f}%",
                        "inline": True,
                    }
                )

            if tp_details:
                tp_lines: list[str] = []
                for i, tp in enumerate(tp_details, 1):
                    if isinstance(tp, dict):
                        tp_price = float(tp.get("price") or tp.get("tp") or 0)
                    else:
                        tp_price = float(tp or 0)
                    if entry_price:
                        if side == "BUY":
                            tp_pct = (tp_price - entry_price) / entry_price * 100.0
                        else:
                            tp_pct = (entry_price - tp_price) / entry_price * 100.0
                    else:
                        tp_pct = 0.0
                    tp_lines.append(f"TP{i}: ${tp_price:.6f} ({tp_pct:.2f}%)")
                embed["fields"].append(
                    {
                        "name": "Take Profit",
                        "value": "\n".join(tp_lines),
                        "inline": False,
                    }
                )

            self._post(self.trade_entries_webhook, {"embeds": [embed]})
        except Exception as e:
            logger.error(f"Error sending trade entry: {e}")

    def send_trade_exit_notification(self, trade_data: dict[str, Any]):
        try:
            symbol = trade_data.get("symbol")
            pnl = float(trade_data.get("pnl") or 0)
            pnl_percent = float(trade_data.get("pnl_percent") or 0)
            exit_reason = (trade_data.get("exit_reason") or "manual").lower()

            reason_label = {
                "sl": "Stop Loss (wybity)",
                "tp": "Take Profit",
                "manual": "Zamknięcie ręczne",
                "position_missing_on_exchange": "Zamknięcie przy synchronizacji",
                "tp_or_sl": "Zamknięcie TP/SL (nieustalony dokładny powód)",
            }.get(exit_reason, exit_reason or "-")

            color = self.colors["profit"] if pnl > 0 else self.colors["loss"]
            embed = {
                "title": f"POSITION CLOSED: {symbol}",
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
                "fields": [
                    {
                        "name": "Wynik",
                        "value": f"PnL: ${pnl:.2f} ({pnl_percent:+.2f}%)",
                        "inline": True,
                    },
                    {"name": "Powód", "value": reason_label, "inline": True},
                ],
            }
            self._post(self.trade_exits_webhook, {"embeds": [embed]})
        except Exception as e:
            logger.error(f"Error sending trade exit: {e}")

    def send_tp_hit_notification(self, tp_data: dict[str, Any]):
        try:
            price = float(tp_data.get("price", 0))
            tp_level = str(tp_data.get("tp_level") or "TP")
            symbol = str(tp_data.get("symbol") or "-")
            footer = "SL moved to BE" if tp_data.get("sl_moved_to_be") else "SL unchanged"
            embed = {
                "title": f"{tp_level} HIT: {symbol}",
                "color": self.colors["success"],
                "fields": [{"name": "Details", "value": f"Price: ${price:.6f}", "inline": True}],
                "footer": {"text": footer},
            }
            # kieruj TP HIT do kanału "exits"
            self._post(self.trade_exits_webhook, {"embeds": [embed]})
        except Exception as e:
            logger.error(f"Error sending TP hit: {e}")

    def send_alert(self, alert_type: str, message: str, details: dict[str, Any] = None):
        alert_configs = {
            "mode_change": {
                "title": "MODE CHANGE",
                "color": self.colors["info"],
                "emoji": "",
            },
            "high_drawdown": {
                "title": "HIGH DRAWDOWN",
                "color": self.colors["warning"],
                "emoji": "",
            },
        }
        config = alert_configs.get(
            alert_type, {"title": "ALERT", "color": self.colors["neutral"], "emoji": ""}
        )
        embed = {
            "title": config["title"],
            "description": message,
            "color": config["color"],
            "timestamp": datetime.utcnow().isoformat(),
        }
        if details:
            embed["fields"] = [
                {"name": str(k), "value": str(v), "inline": True} for k, v in details.items()
            ]
        self._post(self.alerts_webhook, {"embeds": [embed]})

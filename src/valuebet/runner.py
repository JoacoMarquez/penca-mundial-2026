"""Entrypoints CLI del sistema valuebet.

    python -m src.valuebet.runner scan      # soft+sharp → match → ev → staking → ledger → Telegram
    python -m src.valuebet.runner close     # closing lines de abiertas próximas → CLV → learning
    python -m src.valuebet.runner settle    # liquidar abiertas terminadas → pnl → learning → bankroll
    python -m src.valuebet.runner report    # resumen ROI/CLV/segmentos a Telegram
    python -m src.valuebet.runner bankroll --set 1000
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timezone

from src.valuebet import config as vbconfig
from src.valuebet import ev, ledger, learning, notify, sharp
from src.valuebet.books import oddsapi_vb, pinnacle_vb, supermatch
from src.valuebet.matching import match_events
from src.valuebet.staking import apply_caps, kelly_stake
from src.valuebet.types import (
    Opportunity, Suggestion, new_suggestion_id, opportunity_to_dict,
)

log = logging.getLogger(__name__)

DEDUP_MIN_EDGE_GAIN = 0.02

# Criterio de promoción paper → real (mecánico, sin juicio)
PROMOTION_MIN_N_CLV = 50
PROMOTION_MIN_CLV_MEAN = 0.01


# -------------------- scan --------------------

def cmd_scan(cfg: vbconfig.VBConfig) -> None:
    notifier = notify.get_notifier()
    try:
        _scan(cfg, notifier)
    except Exception as e:
        log.exception("scan falló")
        try:
            notifier.send(notify.format_error("scan", e))
        finally:
            raise


def _scan(cfg: vbconfig.VBConfig, notifier) -> None:
    soft = supermatch.fetch_quotes(cfg)
    soft_source = "supermatch"
    if not soft:
        soft = oddsapi_vb.fetch_quotes(cfg)
        soft_source = "oddsapi(fallback)"
    sharp_quotes = pinnacle_vb.fetch_quotes(cfg)
    log.info("scan: %d cuotas soft (%s), %d sharp", len(soft), soft_source, len(sharp_quotes))
    if not soft or not sharp_quotes:
        log.warning("scan: sin datos suficientes (soft=%d sharp=%d) — nada para hacer",
                    len(soft), len(sharp_quotes))
        return

    # matchear evento a evento y armar (cuota_soft, fair_probs, cuota_sharp_outcome)
    pairs = match_events(soft, sharp_quotes, cfg.aliases)
    matched: list[tuple] = []
    for soft_q, sharp_evt_quotes in pairs:
        if soft_q.market not in _allowed_markets(soft_q, cfg):
            continue
        same_market = {q.outcome: q.decimal_odds for q in sharp_evt_quotes
                       if q.market == soft_q.market}
        if soft_q.outcome not in same_market or len(same_market) < 2:
            continue
        try:
            fair = sharp.fair_probs(same_market, soft_q.market, cfg.sharp)
        except ValueError:
            continue
        matched.append((soft_q, fair, same_market[soft_q.outcome]))

    state = learning.load_state()
    singles = ev.find_singles(matched, cfg, state)
    all_s = ledger.load_all()
    bankroll = ledger.load_bankroll(cfg.bankroll_file)
    open_total, open_parlay = ledger.open_exposure(all_s, cfg.mode)

    new_suggestions: list[Suggestion] = []
    for opp in singles:
        s = _build_suggestion([opp], "single", cfg, state, bankroll,
                              open_total, open_parlay)
        if s and not _is_dup(s, all_s):
            new_suggestions.append(s)
            open_total += s.stake_suggested

    for legs in ev.find_parlays(singles, cfg):
        s = _build_suggestion(legs, "parlay", cfg, state, bankroll,
                              open_total, open_parlay)
        if s and not _is_dup(s, all_s):
            new_suggestions.append(s)
            open_total += s.stake_suggested
            open_parlay += s.stake_suggested

    if not new_suggestions:
        log.info("scan: sin oportunidades sobre el umbral")
        return

    ledger.append(new_suggestions)
    for s in new_suggestions:
        notifier.send(notify.format_suggestion(s))
    log.info("scan: %d sugerencias nuevas notificadas", len(new_suggestions))


def _allowed_markets(q, cfg: vbconfig.VBConfig) -> set[str]:
    allowed = set()
    for m in cfg.markets.get(q.sport, []):
        if m == "total":
            if q.market.startswith("total_"):
                allowed.add(q.market)
        else:
            allowed.add(m)
    return allowed


def _build_suggestion(
    legs: list[Opportunity], kind: str, cfg: vbconfig.VBConfig,
    state: learning.LearningState, bankroll: float,
    open_total: float, open_parlay: float,
) -> Suggestion | None:
    c_odds = ev.combined_odds(legs)
    c_fair = ev.combined_fair_prob(legs)
    c_edge = ev.combined_edge(legs)
    # multiplicador de learning: el mínimo entre los segmentos de las patas (conservador)
    mult = min(state.multiplier(o.segment) for o in legs)
    fraction = cfg.kelly_fraction * mult
    stake = kelly_stake(c_fair, c_odds, bankroll, fraction)
    stake = apply_caps(stake, kind, bankroll, open_total, open_parlay, cfg)
    if stake <= 0:
        return None
    return Suggestion(
        id=new_suggestion_id(),
        ts_utc=datetime.now(timezone.utc).isoformat(),
        mode=cfg.mode,
        legs=[opportunity_to_dict(o) for o in legs],
        combined_odds=c_odds, combined_fair_prob=c_fair, edge=c_edge,
        stake_suggested=stake, bankroll_at_ts=bankroll,
        kelly_fraction_used=fraction,
    )


def _is_dup(s: Suggestion, all_s: list[Suggestion]) -> bool:
    keys = {f"{l['quote']['event_id']}|{l['quote']['market']}|{l['quote']['outcome']}"
            for l in s.legs}
    return ledger.is_duplicate(keys, all_s, DEDUP_MIN_EDGE_GAIN, s.edge)


# -------------------- close (CLV) --------------------

def cmd_close(cfg: vbconfig.VBConfig, horizon_minutes: int = 45) -> None:
    """Captura closing lines de sugerencias abiertas cuyo primer evento arranca pronto."""
    all_s = ledger.load_all()
    now = datetime.now(timezone.utc)
    pending = [
        s for s in ledger.open_suggestions(all_s)
        if s.clv is None and any(
            0 <= (_dt(l["quote"]["start_utc"]) - now).total_seconds() <= horizon_minutes * 60
            for l in s.legs
        )
    ]
    if not pending:
        log.info("close: nada próximo a cerrar")
        return

    sharp_quotes = pinnacle_vb.fetch_quotes(cfg)
    by_key = {(q.event_id, q.market, q.outcome): q for q in sharp_quotes}
    state = learning.load_state()
    updated = False

    for s in pending:
        legs_closing = []
        for l in s.legs:
            q = l["quote"]
            # el sharp_event_id quedó en el matching: buscar por nombre+mercado como fallback
            sq = _find_sharp_quote(by_key, sharp_quotes, l)
            if sq is None:
                legs_closing.append({"outcome_key": _leg_key(l), "closing_odds": None,
                                     "closing_fair_prob": None, "clv": None})
                continue
            same_market = {x.outcome: x.decimal_odds for x in sharp_quotes
                           if x.event_id == sq.event_id and x.market == sq.market}
            try:
                fair = sharp.fair_probs(same_market, sq.market, cfg.sharp)
            except ValueError:
                legs_closing.append({"outcome_key": _leg_key(l), "closing_odds": None,
                                     "closing_fair_prob": None, "clv": None})
                continue
            cfp = fair.get(sq.outcome)
            clv_leg = q["decimal_odds"] * cfp - 1.0 if cfp else None
            legs_closing.append({"outcome_key": _leg_key(l), "closing_odds": sq.decimal_odds,
                                 "closing_fair_prob": cfp, "clv": clv_leg})

        ledger.apply_closing(s, legs_closing)
        if s.clv is not None:
            deactivated = learning.update_from_suggestion(state, s)
            updated = True
            for seg in deactivated:
                notify.get_notifier().send(
                    f"⛔ valuebet: segmento <code>{seg}</code> desactivado por CLV negativo sostenido")

    ledger.save_all(all_s)
    if updated:
        learning.save_state(state)
    log.info("close: %d sugerencias procesadas", len(pending))


def _leg_key(l: dict) -> str:
    q = l["quote"]
    return f"{q['event_id']}|{q['market']}|{q['outcome']}"


def _find_sharp_quote(by_key: dict, sharp_quotes, leg: dict):
    q = leg["quote"]
    # 1. match directo si la pata ya era una cuota pinnacle (fallback oddsapi no matchea así)
    direct = by_key.get((q["event_id"], q["market"], q["outcome"]))
    if direct:
        return direct
    # 2. por nombre de evento + mercado + outcome
    from src.valuebet.matching import norm_name
    target = norm_name(q["event_name"])
    for sq in sharp_quotes:
        if (sq.market == q["market"] and sq.outcome == q["outcome"]
                and norm_name(sq.event_name) == target):
            return sq
    return None


# -------------------- settle --------------------

def cmd_settle(cfg: vbconfig.VBConfig, force_results: list[str] | None = None) -> None:
    """Liquida sugerencias. Fase 1: resultados manuales vía --force-result id=won|lost|void.

    La liquidación automática (resultados desde el endpoint de Supermatch o ESPN)
    entra en Fase 2 cuando el recon mapee la fuente.
    """
    if not force_results:
        log.info("settle: sin resultados para aplicar (automático llega en Fase 2; "
                 "usar --force-result <id>=won|lost|void)")
        return
    all_s = ledger.load_all()
    state = learning.load_state()
    notifier = notify.get_notifier()
    bankroll = ledger.load_bankroll(cfg.bankroll_file)

    for spec in force_results:
        sid, _, result = spec.partition("=")
        s = ledger.find(sid, all_s)
        if s is None or s.status != "open":
            log.warning("settle: %s no existe o no está abierta", sid)
            continue
        ledger.settle(s, result)
        # bankroll solo se mueve con apuestas tomadas en modo real
        if s.mode == "real" and s.taken:
            bankroll += s.pnl
        notifier.send(notify.format_settlement(s))

    ledger.save_all(all_s)
    ledger.save_bankroll(bankroll, cfg.bankroll_file)
    learning.save_state(state)


# -------------------- report --------------------

def cmd_report(cfg: vbconfig.VBConfig, send: bool = True) -> dict:
    all_s = ledger.load_all()
    state = learning.load_state()
    settled = [s for s in all_s if s.status in ("won", "lost") and s.pnl is not None]
    with_clv = [s for s in all_s if s.clv is not None]
    staked = sum(s.stake_suggested for s in settled)

    seg_stats: dict[str, dict] = {}
    for seg, d in state.segments.items():
        seg_stats[seg] = {**d, "clv_mean": (d["sum_clv"] / d["n"]) if d["n"] else 0.0}

    clv_mean = (sum(s.clv for s in with_clv) / len(with_clv)) if with_clv else 0.0
    sports_pos = _sports_with_positive_clv(with_clv)
    promotion_ready = (
        cfg.mode == "paper"
        and len(with_clv) >= PROMOTION_MIN_N_CLV
        and clv_mean > PROMOTION_MIN_CLV_MEAN
        and sports_pos >= min(2, len(cfg.sports))
    )

    stats = {
        "mode": cfg.mode,
        "bankroll": ledger.load_bankroll(cfg.bankroll_file),
        "n_total": len(all_s),
        "n_open": len(ledger.open_suggestions(all_s)),
        "n_settled": len(settled),
        "pnl_total": sum(s.pnl for s in settled),
        "roi": (sum(s.pnl for s in settled) / staked) if staked else 0.0,
        "n_clv": len(with_clv),
        "clv_mean": clv_mean,
        "segments": seg_stats,
        "promotion_ready": promotion_ready,
    }
    if send:
        notify.get_notifier().send(notify.format_report(stats))
    return stats


def _sports_with_positive_clv(with_clv: list[Suggestion]) -> int:
    by_sport: dict[str, list[float]] = defaultdict(list)
    for s in with_clv:
        for l in s.legs:
            by_sport[l["quote"]["sport"]].append(s.clv)
    return sum(1 for vals in by_sport.values() if vals and sum(vals) / len(vals) > 0)


# -------------------- util --------------------

def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="valuebet")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan")
    sub.add_parser("close")
    p_settle = sub.add_parser("settle")
    p_settle.add_argument("--force-result", action="append", metavar="ID=RESULT",
                          help="liquidación manual: <id>=won|lost|void|push")
    sub.add_parser("report")
    p_bank = sub.add_parser("bankroll")
    p_bank.add_argument("--set", type=float, dest="amount", required=True)

    args = ap.parse_args()
    cfg = vbconfig.load()

    if args.cmd == "scan":
        cmd_scan(cfg)
    elif args.cmd == "close":
        cmd_close(cfg)
    elif args.cmd == "settle":
        cmd_settle(cfg, args.force_result)
    elif args.cmd == "report":
        cmd_report(cfg)
    elif args.cmd == "bankroll":
        ledger.save_bankroll(args.amount, cfg.bankroll_file)
        print(f"bankroll seteado: {args.amount:.0f} UYU")


if __name__ == "__main__":
    main()

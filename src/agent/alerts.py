"""Banderas rojas (red-flags) automáticas + digest de jornada.

Codifican los errores YA conocidos para que no se repitan callados. NO anticipan
errores nuevos — eso lo sigue cazando un humano. Dos grupos:

PRE-PARTIDO (corren en el pipeline, cuando todavía se puede actuar):
    - concentración: demasiadas planillas en un mismo marcador (caza el flood de m108).
    - lambda_vs_xi: Capa 4 bajó el λ citando a un jugador que está en el XI confirmado
      (caza el caso Enciso).

POST-JORNADA (corren en el digest, para aprender):
    - pool_slip: nuestra mejor penca se desplomó en el pool.
    - llm_backfire: un ajuste de Capa 4 terminó empeorando la predicción.

Todo es de monitoreo: no toca estrategia, modelo ni probabilidad de ganar.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# --- Umbrales (acordados con el usuario; tunables) ---
CONCENTRATION_MAX_SHARE = 0.40   # >40% de las picks en un marcador → bandera
CONCENTRATION_MIN_DISTINCT = 4   # o menos de 4 marcadores distintos
LAMBDA_CUT_THRESHOLD = -0.10     # Δλ ≤ esto se considera "recorte significativo"
POOL_SLIP_POSITIONS = 25         # caída de más de N puestos → bandera
LLM_BACKFIRE_DELTA = 0.15        # |Δλ| ≥ esto + helpful="no" → bandera


@dataclass(frozen=True)
class Flag:
    level: str   # "warn" | "info"
    code: str    # concentration | lambda_xi | pool_slip | llm_backfire
    title: str
    detail: str


# ============ PRE-PARTIDO ============

def check_concentration(scores: list, n_pencas: int | None = None) -> Flag | None:
    """`scores`: lista de (gL, gV) — los marcadores asignados a las N planillas."""
    scores = [tuple(s) for s in scores if s and s[0] is not None]
    n = n_pencas or len(scores)
    if n < 5 or not scores:
        return None
    exp = Counter(scores)
    top_score, top_count = exp.most_common(1)[0]
    distinct = len(exp)
    share = top_count / n
    if share > CONCENTRATION_MAX_SHARE or distinct < CONCENTRATION_MIN_DISTINCT:
        return Flag(
            "warn", "concentration",
            "Concentración alta de picks",
            f"{top_count}/{n} planillas en {top_score[0]}-{top_score[1]} "
            f"({share:.0%}); solo {distinct} marcadores distintos.",
        )
    return None


def _surnames(xi: list) -> list[str]:
    """Apellidos (último token, ≥4 letras) de los nombres del XI, para matchear en texto."""
    out = []
    for name in xi or []:
        if not name:
            continue
        toks = [t for t in re.split(r"\s+", str(name).strip()) if len(t) >= 4]
        if toks:
            out.append(toks[-1])
    return out


def check_lambda_vs_xi(
    qa: dict | None,
    home_xi: list | None,
    away_xi: list | None,
    home_team: str = "local",
    away_team: str = "visitante",
) -> Flag | None:
    """Capa 4 bajó el λ de un equipo citando a un jugador que SÍ está en el XI confirmado.

    Heurística: matchea apellidos del XI contra el texto del reasoning. Puede tener falsos
    positivos/negativos — es una bandera de "revisá esto", no un veredicto.
    """
    if not qa:
        return None
    reasoning = (qa.get("reasoning") or "").lower()
    if not reasoning:
        return None
    hits = []
    for side, delta_key, xi, team in (
        ("local", "delta_lambda_L", home_xi, home_team),
        ("visitante", "delta_lambda_V", away_xi, away_team),
    ):
        delta = float(qa.get(delta_key) or 0.0)
        if delta > LAMBDA_CUT_THRESHOLD:   # no recortó significativamente este lado
            continue
        for surname in _surnames(xi):
            if surname.lower() in reasoning:
                hits.append((team, surname, delta))
                break
    if not hits:
        return None
    parts = [f"{team}: bajó λ {delta:+.2f} citando a «{sn}», pero está en el XI"
             for team, sn, delta in hits]
    return Flag(
        "warn", "lambda_xi",
        "Capa 4 contradice el XI confirmado",
        "; ".join(parts) + ". Revisá si el ajuste corresponde.",
    )


# ============ POST-JORNADA ============

def check_pool_slippage(
    prev_entries: list | None,
    curr_entries: list | None,
    my_ids: set,
) -> Flag | None:
    """Nuestra mejor penca (por puntos acumulados) cayó >N puestos, o quedó bajo la mediana."""
    if not curr_entries:
        return None

    def rank_map(entries):
        s = sorted(entries, key=lambda e: -int(e.get("points_total", 0)))
        return {int(e.get("penca_id", 0)): i + 1 for i, e in enumerate(s)}, s

    curr_rank, curr_sorted = rank_map(curr_entries)
    mine = [e for e in curr_entries if int(e.get("penca_id", 0)) in my_ids]
    if not mine:
        return None
    best = max(mine, key=lambda e: int(e.get("points_total", 0)))
    bid = int(best.get("penca_id", 0))
    pts = int(best.get("points_total", 0))
    cur_pos = curr_rank.get(bid)

    median_pts = int(curr_sorted[len(curr_sorted) // 2].get("points_total", 0)) if curr_sorted else 0

    drop = None
    if prev_entries:
        prev_rank, _ = rank_map(prev_entries)
        prev_pos = prev_rank.get(bid)
        if prev_pos is not None and cur_pos is not None:
            drop = cur_pos - prev_pos

    if (drop is not None and drop > POOL_SLIP_POSITIONS) or pts < median_pts:
        bits = [f"mejor penca {cur_pos}° del pool con {pts} pts"]
        if drop is not None and drop > POOL_SLIP_POSITIONS:
            bits.append(f"cayó {drop} puestos")
        if pts < median_pts:
            bits.append(f"bajo la mediana ({median_pts} pts)")
        return Flag("warn", "pool_slip", "Caída en el pool", " · ".join(bits) + ".")
    return None


def check_llm_backfired(report: dict) -> Flag | None:
    """Un ajuste de Capa 4 con magnitud real terminó empeorando la predicción."""
    if (report.get("llm_adjustment_was_helpful") or "") != "no":
        return None
    qa = report.get("llm_adjustment_applied") or {}
    mag = max(abs(float(qa.get("delta_lambda_L") or 0)), abs(float(qa.get("delta_lambda_V") or 0)))
    if mag < LLM_BACKFIRE_DELTA:
        return None
    return Flag(
        "info", "llm_backfire",
        "Ajuste de Capa 4 contraproducente",
        f"{report.get('home_team','?')} vs {report.get('away_team','?')}: "
        f"el ajuste del LLM (Δλ máx {mag:.2f}) fue en contra del resultado.",
    )


# ============ render para Telegram ============

def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_flags(flags: list[Flag]) -> str:
    """Texto HTML de un conjunto de banderas (para el alerta o el digest)."""
    icon = {"warn": "🚩", "info": "ℹ️"}
    lines = []
    for f in flags:
        lines.append(f"{icon.get(f.level,'•')} <b>{_esc(f.title)}</b>\n   {_esc(f.detail)}")
    return "\n".join(lines)


def build_digest_text(
    date_label: str,
    match_summaries: list[dict],
    flags: list[Flag],
    pool_line: str | None = None,
) -> str:
    """Digest de una jornada: resumen de resultados + banderas post-jornada.

    match_summaries: [{label, final, best_pts, best_score}].
    """
    sep = "━━━━━━━━━━━━━━━"
    lines = [f"📅 <b>Jornada {_esc(date_label)}</b>"]
    if pool_line:
        lines.append(pool_line)
    lines.append(sep)
    for s in match_summaries:
        lines.append(
            f"⚽ {_esc(s['label'])}  <b>{_esc(s['final'])}</b>  ·  "
            f"mejor: {s['best_pts']} pts ({_esc(s['best_score'])})"
        )
    lines.append(sep)
    if flags:
        lines.append("<b>🚩 Para revisar</b>")
        lines.append(render_flags(flags))
    else:
        lines.append("✅ Sin banderas — jornada limpia.")
    return "\n".join(lines)

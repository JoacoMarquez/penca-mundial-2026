"""Pipeline de picks para el flujo MANUAL de la Penca Supermatch.

Decisión operativa 2026-08-04: la carga en la web la hace el usuario a mano (menos
superficie de riesgo con la cuenta; son 8 partidos por semana). Este módulo hace
todo lo demás:

    1. Refresca fixture/resultados desde el penca-api público.
    2. Ajusta ratings con TODO el histórico disponible (temporadas previas + partidos
       ya jugados del Clausura).
    3. Trae odds de Supermatch ES para los partidos que ya tienen mercado y ajusta λ
       (blend 70% mercado / 30% ratings, mismo criterio que la Capa 2 del Mundial).
    4. Calibra el pool con la tasa de exactos del ranking público (si ya hay fechas
       jugadas).
    5. Optimiza el portfolio de la TEMPORADA completa con lo ya cargado congelado,
       y emite la planilla de la fecha objetivo.
    6. Versiona en data/predictions/clausura/ y (opcional) manda la planilla por
       Telegram lista para copiar en la web.

Uso:
    python -m src.clausura.picks --fecha 1                  # planilla de la Fecha 1
    python -m src.clausura.picks --fecha 1 --telegram       # y la manda al bot
    python -m src.clausura.picks --fecha 1 --participaciones 8
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.clausura.api import TZ_UY, PencaApiClient
from src.clausura.economics import MAX_GOALS, SimConfig, index_score, score_index
from src.clausura.historical import DATA_DIR, load_dataset
from src.clausura.odds import EventOdds, _norm, fetch_primera_odds
from src.clausura.pool import PoolConfig, calibrate_from_exact_rate, observed_exact_rate_from_ranking
from src.clausura.especiales import (
    fetch_opciones,
    goleador_prior_from_ratings,
    p_campeon_from_grids,
    pool_campeon_distribution,
    pool_goleador_distribution,
)
from src.clausura.ratings import TeamRatings, fit_ratings
from src.clausura.strategy import EspecialesInput, PortfolioClausura, build_portfolio
from src.model.market_probs import devig
from src.model.poisson import MarketConstraints, fit_params, score_grid
from src.utils.versions import latest_version

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "clausura2026.yaml"
PRED_DIR = ROOT / "data" / "predictions" / "clausura"

MARKET_WEIGHT = 0.7   # blend λ: 70% mercado / 30% ratings (criterio Capa 1+2 del Mundial)


# -------------------- carga de insumos --------------------

def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def flat_eventos(cfg: dict) -> list[dict]:
    """Todos los eventos de la temporada, ordenados por inicio, con nº de fecha."""
    out = []
    for nombre, f in cfg["fechas"].items():
        n = int(nombre.split()[-1])
        for ev in f["eventos"]:
            out.append({**ev, "fecha_n": n, "fecha_id": f["fecha_id"]})
    out.sort(key=lambda e: e["inicio_utc"])
    return out


def ensure_ratings() -> TeamRatings:
    """Ratings con todo lo jugado: 5 temporadas previas + lo que haya del Clausura."""
    path = DATA_DIR / "primera_uy_historico.json"
    if not path.exists():
        log.info("histórico ausente — descargando del penca-api…")
        from src.clausura.historical import main as hist_main
        import sys
        argv, sys.argv = sys.argv, ["historical"]
        try:
            hist_main()
        finally:
            sys.argv = argv
    return fit_ratings(load_dataset())


def match_odds(eventos: list[dict], odds: list[EventOdds]) -> dict[int, EventOdds]:
    """evento_id → EventOdds, matcheando por nombres normalizados de ambos equipos.

    Los nombres del penca-api y del Elasticsearch pueden diferir en detalles
    ("Montevideo City Torque" vs "Montevideo City"); matcheamos por inclusión de
    tokens normalizados en ambas direcciones.
    """
    def similar(a: str, b: str) -> bool:
        na, nb = _norm(a), _norm(b)
        return na == nb or na in nb or nb in na

    out = {}
    for ev in eventos:
        for o in odds:
            if similar(ev["local"], o.home) and similar(ev["visitante"], o.away):
                out[ev["evento_id"]] = o
                break
    return out


# -------------------- construcción de grillas --------------------

def delta_grid(gl: int, gv: int) -> np.ndarray:
    """Partido ya jugado: toda la masa en el resultado real."""
    g = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    g[min(gl, MAX_GOALS), min(gv, MAX_GOALS)] = 1.0
    return g


def market_lambdas(o: EventOdds) -> tuple[float, float] | None:
    """λ del mercado: fit de Poisson contra 1X2 (+ over 2.5 si está)."""
    if not o.x1x2:
        return None
    p = devig(o.x1x2, "proportional")
    o25 = devig(o.totals["2.5"], "proportional").get("over") if "2.5" in o.totals else None
    c = MarketConstraints(p_home_win=p["home"], p_draw=p["draw"], p_away_win=p["away"],
                          p_over_2_5=o25)
    lam_l, lam_v, _ = fit_params(c)
    return lam_l, lam_v


def build_season_grids(
    eventos: list[dict],
    ratings: TeamRatings,
    odds_by_evento: dict[int, EventOdds],
    resultados: dict[int, tuple[int, int]],
) -> tuple[list[np.ndarray], list[str]]:
    """Grilla por evento + etiqueta de la fuente usada (para el reporte)."""
    grids, fuentes = [], []
    for ev in eventos:
        eid = ev["evento_id"]
        if eid in resultados:
            gl, gv = resultados[eid]
            grids.append(delta_grid(gl, gv))
            fuentes.append(f"final {gl}-{gv}")
            continue

        lam_rt = ratings.lambdas(ev["local"], ev["visitante"])
        o = odds_by_evento.get(eid)
        lam_mkt = market_lambdas(o) if o else None
        if lam_mkt:
            lam_l = MARKET_WEIGHT * lam_mkt[0] + (1 - MARKET_WEIGHT) * lam_rt[0]
            lam_v = MARKET_WEIGHT * lam_mkt[1] + (1 - MARKET_WEIGHT) * lam_rt[1]
            fuentes.append("mercado+ratings")
        else:
            lam_l, lam_v = lam_rt
            fuentes.append("ratings")
        grids.append(score_grid(lam_l, lam_v, 0.0, max_goals=MAX_GOALS))
    return grids, fuentes


# -------------------- estado congelado (picks ya cargados) --------------------

def fecha_dir(n: int) -> Path:
    return PRED_DIR / f"fecha_{n:02d}"


def load_frozen(
    eventos: list[dict],
    target_fecha: int,
    n_participaciones: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(frozen_picks, frozen_mask) desde los archivos versionados de fechas anteriores.

    Un partido queda congelado si pertenece a una fecha anterior a la objetivo Y hay
    picks guardados para él. Si falta el archivo de una fecha pasada avisamos: eso
    significa que lo cargado en la web no está registrado acá.
    """
    n = len(eventos)
    frozen = np.zeros((n_participaciones, n), dtype=np.int64)
    mask = np.zeros(n, dtype=bool)
    idx_by_evento = {ev["evento_id"]: i for i, ev in enumerate(eventos)}

    for f in range(1, target_fecha):
        d = fecha_dir(f)
        latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
        if latest is None:
            log.warning("fecha %d sin picks guardados — sus partidos quedan libres "
                        "(si ya los cargaste en la web, hay drift)", f)
            continue
        data = json.loads(latest.read_text(encoding="utf-8"))
        for row in data["picks"]:
            i = idx_by_evento.get(row["evento_id"])
            if i is None:
                continue
            for k, (gl, gv) in enumerate(row["scores"][:n_participaciones]):
                frozen[k, i] = score_index(gl, gv)
            mask[i] = True
    return frozen, mask


def load_frozen_especiales(
    target_fecha: int,
    n_participaciones: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Campeón/Goleador ya guardados en cualquier corrida anterior (-1 = libre).

    Los especiales se eligen una vez (idealmente antes de la Fecha 1); si un archivo
    previo los tiene, se asumen cargados en la web y quedan fijos.
    """
    for f in range(target_fecha, 0, -1):
        d = fecha_dir(f)
        latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
        if latest is None:
            continue
        data = json.loads(latest.read_text(encoding="utf-8"))
        esp = data.get("especiales")
        if not esp:
            continue
        campeon = np.full(n_participaciones, -1, dtype=np.int64)
        goleador = np.full(n_participaciones, -1, dtype=np.int64)
        for i, row in enumerate(esp.get("por_participacion", [])[:n_participaciones]):
            campeon[i] = row.get("campeon_idx", -1)
            goleador[i] = row.get("goleador_idx", -1)
        return campeon, goleador
    return None, None


def format_especiales(port: PortfolioClausura, equipo_nombres: list[str],
                      opciones_goleador) -> str:
    """Sección de la planilla con Campeón/Goleador por participación."""
    if port.campeon is None:
        return ""
    lines = ["<b>⭐ Especiales (25 pts c/u — pestaña Especial)</b>"]
    if port.p_campeon is not None:
        top = np.argsort(-port.p_campeon)[:4]
        probs = " · ".join(f"{equipo_nombres[t]} {port.p_campeon[t]:.0%}" for t in top)
        lines.append(f"<i>P(campeón): {probs}</i>")
    for i in range(len(port.campeon)):
        campeon = equipo_nombres[int(port.campeon[i])]
        gol = ""
        if port.goleador is not None and opciones_goleador:
            gol = f" · goleador: {opciones_goleador[int(port.goleador[i])].nombre}"
        lines.append(f"  {i + 1}: campeón <b>{campeon}</b>{gol}")
    if port.goleador is None:
        lines.append("  <i>goleador: menú aún no publicado por Supermatch — "
                     "recomiendo cargarlo apenas aparezca</i>")
    return "\n".join(lines)


def save_version(target_fecha: int, payload: dict) -> Path:
    """Escribe v<N>_<ts>.json sin sobreescribir (regla de trabajo #2)."""
    d = fecha_dir(target_fecha)
    d.mkdir(parents=True, exist_ok=True)
    prev = latest_version(d.glob("v*_*.json"))
    n = 1
    if prev is not None:
        from src.utils.versions import version_num
        n = version_num(prev) + 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = d / f"v{n}_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# -------------------- planilla --------------------

def format_planilla(
    eventos_fecha: list[dict],
    portfolio: PortfolioClausura,
    idx_of: dict[int, int],
    fuentes: list[str],
    n_rivales: int,
) -> str:
    """Planilla en texto plano/HTML-Telegram, ordenada como la web (por kickoff)."""
    r = portfolio.resultado
    lines = []
    lines.append(f"<b>🏆 Penca Clausura — Fecha {eventos_fecha[0]['fecha_n']}</b>")
    lines.append(f"E[premio] portfolio: ${r.e_premio_total:,.0f} · "
                 f"P(premio grande): {r.p_gana_penca:.0%} · pool: {n_rivales}")
    lines.append("")
    for ev in eventos_fecha:
        i = idx_of[ev["evento_id"]]
        estrella = " ⭐x2" if ev["preferencial"] else ""
        cierre = datetime.fromisoformat(ev["cierre_pronostico_utc"]).astimezone(TZ_UY)
        cierre_uy = cierre.strftime("%d/%m %H:%M")
        scores = [index_score(int(portfolio.picks[k, i]))
                  for k in range(portfolio.picks.shape[0])]
        marcadores = " | ".join(f"{a}-{b}" for a, b in scores)
        lines.append(f"<b>{ev['local']} vs {ev['visitante']}</b>{estrella}")
        lines.append(f"  {marcadores}")
        lines.append(f"  <i>cierra {cierre_uy} UY · {fuentes[i]}</i>")
    lines.append("")
    lines.append("Orden de participaciones: 1=EV puro, 2..N=perturbaciones óptimas.")
    lines.append("Cargar en supermatch.com.uy → Pencas → Torneo Clausura 2026.")
    return "\n".join(lines)


# -------------------- main --------------------

def run(
    target_fecha: int,
    n_participaciones: int = 5,
    telegram: bool = False,
    n_sims: int = 800,
) -> Path:
    cfg = load_config()
    eventos = flat_eventos(cfg)
    idx_of = {ev["evento_id"]: i for i, ev in enumerate(eventos)}

    # resultados ya conocidos + tamaño real del pool + tasa de exactos
    resultados: dict[int, tuple[int, int]] = {}
    n_rivales, exact_rate = 151, None
    penca_id = cfg["pencas"]["paga"]["id"]
    try:
        with PencaApiClient() as api:
            # resultados de partidos finalizados
            for nombre, f in cfg["fechas"].items():
                data = api._get(f"/front/campeonatos/fechas/{f['fecha_id']}/eventos")
                for e in data:
                    res = e.get("resultado") or {}
                    gl, gv = res.get("golesEquipoLocal"), res.get("golesEquipoVisitante")
                    if gl is not None and gv is not None:
                        resultados[e["id"]] = (int(gl), int(gv))
            ranking = api.ranking(penca_id)
            n_rivales = max(len(ranking), 1)
            jugados = len(resultados)
            exact_rate = observed_exact_rate_from_ranking(ranking, jugados)
    except Exception as e:  # red caída: seguimos con defaults
        log.warning("penca-api no disponible (%s) — sigo con defaults", e)

    ratings = ensure_ratings()

    try:
        odds = fetch_primera_odds()
    except Exception as e:
        log.warning("odds de Supermatch no disponibles (%s)", e)
        odds = []
    odds_by_evento = match_odds(eventos, odds)

    grids, fuentes = build_season_grids(eventos, ratings, odds_by_evento, resultados)

    pool_cfg = PoolConfig()
    if exact_rate is not None:
        jugables = [g for g, ev in zip(grids, eventos) if ev["evento_id"] in resultados]
        if jugables:
            pool_cfg = calibrate_from_exact_rate(jugables, exact_rate, pool_cfg)
            log.info("pool calibrado: temperatura=%.2f (exact_rate=%.3f)",
                     pool_cfg.temperature, exact_rate)

    frozen, mask = load_frozen(eventos, target_fecha, n_participaciones)
    # los partidos ya jugados también quedan fijos (su pick ya no se puede cambiar)
    for i, ev in enumerate(eventos):
        if ev["evento_id"] in resultados and not mask[i]:
            mask[i] = True  # sin picks guardados: quedan en 0-0; no afecta el futuro

    # ---------- pool: prior + Q empírica del snapshot (si el campeonato ya inició) ----------
    from src.clausura.pool import pool_distribution
    from src.clausura.pool_snapshot import (
        blended_q, empirical_campeon_counts, empirical_counts, load_latest_snapshot,
    )
    pool_qs = [pool_distribution(g, pool_cfg) for g in grids]
    snapshot = load_latest_snapshot(max_age_hours=48)
    campeon_counts = None
    if snapshot:
        counts = empirical_counts(snapshot)
        observados = 0
        for i, ev in enumerate(eventos):
            c = counts.get(ev["evento_id"])
            if c is not None and c.sum() > 0:
                pool_qs[i] = blended_q(pool_qs[i], c)
                observados += 1
        log.info("pool empírico: snapshot de %s participaciones, %d/%d eventos observados",
                 snapshot.get("n_participaciones"), observados, len(eventos))
        campeon_counts = snapshot  # se resuelve más abajo, con el índice de equipos armado

    # ---------- especiales: Campeón (siempre modelable) + Goleador (si hay menú) ----------
    equipos_cfg: dict[int, str] = cfg["equipos"]
    equipo_nombres = [equipos_cfg[k] for k in sorted(equipos_cfg)]
    equipo_idx = {nombre: i for i, nombre in enumerate(equipo_nombres)}
    local_de = np.array([equipo_idx[ev["local"]] for ev in eventos])
    visita_de = np.array([equipo_idx[ev["visitante"]] for ev in eventos])

    p_champ_prior = p_campeon_from_grids(grids, local_de, visita_de, len(equipo_nombres))
    pool_q_campeon = pool_campeon_distribution(p_champ_prior, equipo_nombres)
    if campeon_counts is not None:
        c = empirical_campeon_counts(campeon_counts, equipo_idx, len(equipo_nombres))
        pool_q_campeon = blended_q(pool_q_campeon, c)

    opciones_campeon, opciones_goleador = None, None
    try:
        opciones_campeon, opciones_goleador = fetch_opciones(penca_id)
    except Exception as e:
        log.warning("opciones de especiales no disponibles (%s)", e)

    p_gol, pool_q_gol = None, None
    if opciones_goleador:
        equipo_id_por_nombre = {v: k for k, v in equipos_cfg.items()}
        p_gol = goleador_prior_from_ratings(
            opciones_goleador, equipo_nombres, equipo_id_por_nombre, ratings.ataque
        )
        pool_q_gol = pool_goleador_distribution(p_gol)
    else:
        log.info("goleador: Supermatch aún no publicó el menú de opciones (500) — "
                 "solo se recomienda Campeón por ahora")

    frozen_campeon, frozen_goleador = load_frozen_especiales(target_fecha, n_participaciones)

    especiales = EspecialesInput(
        local_de=local_de,
        visita_de=visita_de,
        n_teams=len(equipo_nombres),
        pool_q_campeon=pool_q_campeon,
        p_goleador=p_gol,
        pool_q_goleador=pool_q_gol,
        frozen_campeon=frozen_campeon,
        frozen_goleador=frozen_goleador,
    )

    port = build_portfolio(
        grids=grids,
        fecha_de_partido=[ev["fecha_id"] for ev in eventos],
        preferencial=[ev["preferencial"] for ev in eventos],
        n_participaciones=n_participaciones,
        pool_cfg=pool_cfg,
        sim=SimConfig(n_sims=n_sims, n_rivales=n_rivales),
        frozen_picks=frozen,
        frozen_mask=mask,
        especiales=especiales,
        pool_qs=pool_qs,
    )

    eventos_fecha = [ev for ev in eventos if ev["fecha_n"] == target_fecha]
    planilla = format_planilla(eventos_fecha, port, idx_of, fuentes, n_rivales)
    planilla += "\n" + format_especiales(port, equipo_nombres, opciones_goleador)

    payload = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "fecha": target_fecha,
        "n_participaciones": n_participaciones,
        "pool": {"n_rivales": n_rivales, "temperatura": pool_cfg.temperature,
                 "exact_rate_observada": exact_rate},
        "resultado_sim": {
            "e_premio_total": port.resultado.e_premio_total,
            "e_premio_penca": port.resultado.e_premio_penca,
            "e_premio_fechas": port.resultado.e_premio_fechas,
            "p_gana_penca": port.resultado.p_gana_penca,
        },
        "especiales": {
            "p_campeon": {equipo_nombres[i]: round(float(p), 4)
                          for i, p in enumerate(port.p_campeon)
                          if p > 0.001} if port.p_campeon is not None else None,
            "por_participacion": [
                {
                    "campeon_idx": int(port.campeon[i]),
                    "campeon": equipo_nombres[int(port.campeon[i])],
                    "goleador_idx": int(port.goleador[i]) if port.goleador is not None else -1,
                    "goleador": (opciones_goleador[int(port.goleador[i])].nombre
                                 if port.goleador is not None and opciones_goleador else None),
                }
                for i in range(n_participaciones)
            ] if port.campeon is not None else [],
        },
        "picks": [
            {
                "evento_id": ev["evento_id"],
                "partido": f"{ev['local']} vs {ev['visitante']}",
                "preferencial": ev["preferencial"],
                "cierre_pronostico_utc": ev["cierre_pronostico_utc"],
                "fuente_modelo": fuentes[idx_of[ev["evento_id"]]],
                "scores": [list(index_score(int(port.picks[k, idx_of[ev["evento_id"]]])))
                           for k in range(n_participaciones)],
            }
            for ev in eventos_fecha
        ],
    }
    path = save_version(target_fecha, payload)

    print(planilla.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    print(f"\nguardado: {path}")

    if telegram:
        from src.notifier.telegram import TelegramConfig, TelegramNotifier
        TelegramNotifier(TelegramConfig.from_env()).send(planilla)
        print("enviado por Telegram ✓")

    return path


def resolve_fecha(arg: str) -> int:
    """'auto' → primera fecha con partidos pendientes (para el timer del VPS)."""
    if arg.lower() == "auto":
        from src.clausura.dashboard_loader import fecha_actual
        return fecha_actual(load_config())
    return int(arg)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", default="auto",
                    help="número de fecha (1-15) o 'auto' = próxima con partidos pendientes")
    ap.add_argument("--participaciones", type=int, default=5)
    ap.add_argument("--telegram", action="store_true", help="mandar la planilla al bot")
    ap.add_argument("--sims", type=int, default=800)
    args = ap.parse_args()
    run(resolve_fecha(args.fecha), args.participaciones, args.telegram, args.sims)


if __name__ == "__main__":
    main()

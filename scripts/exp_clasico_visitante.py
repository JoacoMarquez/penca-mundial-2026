"""¿Vale cubrir la victoria del NO-favorito en el preferencial ×2 de la fecha?

Motivación (análisis pre-F4, 2026-08-28): las 12 filas juegan Peñarol o empate en
el clásico (L8 E4 V0) con el modelo dando 27% a Nacional. Si gana Nacional, la
media de la planilla es 1.2-1.5 pts sobre 16. El optimizador eligió no cubrir V y
los rechazos de diferenciación están re-validados por el PIT del pool (17/8) — pero
esta hipótesis puntual nunca se midió sola, y el ×2 la hace la más cara de errar.

Qué mide: para CADA fila, el Δ E[premio] pareado de cambiarle SOLO el pick del
partido preferencial de la fecha a un marcador dado (default 0-1), contra la
planilla VIGENTE (la cadena de warm start, lo mismo que audita cold_check — no el
portfolio re-optimizado de esta corrida, que trae churn).

Harness: el mismo del gate por valor (strategy.EvaluadorPortfolio): sorteos
comunes, semillas de evaluación independientes de la optimización, y el doble
umbral del rerun/cold_check (Δ > 2·SE y > $2.000). Con una diferencia de costo:
las N variantes comparten los simuladores por semilla (construir el simulador es
lo caro — 65 s a 2.400 sims; re-cargar picks es barato), que es el patrón
sancionado por `comparar` (dos `_cargar` sobre el mismo simulador).

Winner's curse: mirar el mejor de 12 Δ infla el falso positivo del doble umbral,
así que el mejor candidato se RE-MIDE en semillas frescas (offset aparte) y el
veredicto final sale de esa confirmación, no del barrido.

Solo imprime: no versiona planillas, no manda Telegram, no toca el estado.

Uso (VPS, fuera de la ventana de timers 11:00-23:50 UTC idealmente):
    /opt/penca/.venv/bin/python -m scripts.exp_clasico_visitante --fecha 4
    ... --marcador 0-1 --marcador 1-2      # más de un marcador candidato
"""

from __future__ import annotations

import argparse
import gc
import logging

import numpy as np

log = logging.getLogger(__name__)

UMBRAL_SE = 2.0        # mismos umbrales que rerun_cierre.vale_avisar / cold_check
UMBRAL_ABS = 2_000.0
# Offset extra para las semillas de confirmación: disjuntas de las del barrido
# (EVAL_SEED_OFFSET + 0..seeds-1) y de las de cualquier gate que corra hoy.
CONFIRM_OFFSET = 1_000


def parse_marcador(s: str) -> tuple[int, int]:
    a, b = s.split("-")
    return int(a), int(b)


def matriz_vigente(contexto: dict, target_fecha: int, n_participaciones: int) -> np.ndarray:
    """La planilla VIGENTE (cadena de warm start), huecos rellenos con la corrida.

    Mismo criterio que cold_check.matriz_vigente: lo cargado/por cargar es la
    cadena, no el portfolio churn-eado de esta corrida.
    """
    from src.clausura.picks import load_warm_start

    fria = np.asarray(contexto["portfolio"].picks, dtype=np.int64)
    warm = load_warm_start(contexto["eventos"], target_fecha, n_participaciones)
    if warm is None:
        log.warning("sin planilla previa: la base es el portfolio de esta corrida")
        return fria
    return np.where(warm >= 0, warm, fria)


def col_preferencial(eventos: list[dict], target_fecha: int, evento_id: int | None) -> int:
    for i, ev in enumerate(eventos):
        if evento_id is not None:
            if ev["evento_id"] == evento_id:
                return i
        elif ev["fecha_n"] == target_fecha and ev.get("preferencial"):
            return i
    raise SystemExit("no encontré el partido objetivo (¿--evento-id?)")


def medir(ev, base: np.ndarray, variantes: list[np.ndarray],
          seeds: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
    """Δ pareado de cada variante vs base, compartiendo simulador por semilla.

    Devuelve (delta_medio[i], se[i], valor_base_medio).
    """
    deltas = np.zeros((len(seeds), len(variantes)))
    v_base = []
    for j, seed in enumerate(seeds):
        gc.collect()                      # el simulador es el pico de RAM del VPS
        s = ev._simulador(seed)
        vb = ev._cargar(s, base)
        v_base.append(vb)
        for i, var in enumerate(variantes):
            deltas[j, i] = ev._cargar(s, var) - vb
        log.info("semilla %d/%d liquidada (base $%.0f)", j + 1, len(seeds), vb)
        del s
    se = (deltas.std(axis=0, ddof=1) / np.sqrt(len(seeds))
          if len(seeds) > 1 else np.zeros(len(variantes)))
    return deltas.mean(axis=0), se, float(np.mean(v_base))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", type=int, default=None, help="default: la fecha en curso")
    ap.add_argument("--evento-id", type=int, default=None,
                    help="default: el preferencial de la fecha")
    ap.add_argument("--marcador", action="append", default=None,
                    help="marcador candidato gL-gV (repetible; default 0-1)")
    ap.add_argument("--participaciones", type=int, default=None)
    ap.add_argument("--sims", type=int, default=None,
                    help="default: los mismos sorteos que producción")
    ap.add_argument("--seeds", type=int, default=5, help="semillas del barrido")
    ap.add_argument("--confirm-seeds", type=int, default=5,
                    help="semillas frescas para confirmar el mejor Δ (0 = sin confirmar)")
    args = ap.parse_args()

    from src.clausura.economics import index_score, score_index
    from src.clausura.picks import DEFAULT_SIMS, resolve_fecha
    from src.clausura.picks import run as picks_run
    from src.clausura.rivals import mis_numeros_env
    from src.clausura.strategy import EVAL_SEED_OFFSET

    fecha = args.fecha if args.fecha is not None else resolve_fecha("auto")
    numeros = mis_numeros_env()
    n_part = args.participaciones or len(numeros) or 12
    n_sims = args.sims or DEFAULT_SIMS
    marcadores = [parse_marcador(m) for m in (args.marcador or ["0-1"])]

    contexto: dict = {}
    log.info("pipeline fecha %d, %d participaciones, %d sorteos (solo para armar "
             "el evaluador — no versiona)", fecha, n_part, n_sims)
    picks_run(fecha, n_part, telegram=False, n_sims=n_sims,
              contexto=contexto, usar_warm_start=True, guardar=False)

    ev = contexto.get("evaluador")
    if ev is None:
        raise SystemExit("la corrida no dejó evaluador en el contexto")

    base = matriz_vigente(contexto, fecha, n_part)
    col = col_preferencial(contexto["eventos"], fecha, args.evento_id)
    partido = contexto["eventos"][col]
    log.info("partido objetivo: %s vs %s (evento %d)%s",
             partido["local"], partido["visitante"], partido["evento_id"],
             " ×2" if partido.get("preferencial") else "")

    variantes, etiquetas = [], []
    for gl, gv in marcadores:
        idx = score_index(gl, gv)
        for k in range(n_part):
            if base[k, col] == idx:
                continue
            var = base.copy()
            var[k, col] = idx
            variantes.append(var)
            actual = index_score(int(base[k, col]))
            etiquetas.append((k, f"{gl}-{gv}", f"{actual[0]}-{actual[1]}"))
    if not variantes:
        raise SystemExit("todas las filas ya juegan ese marcador — nada que medir")

    seed0 = ev._cfg.seed + EVAL_SEED_OFFSET
    delta, se, v_base = medir(ev, base, variantes, [seed0 + k for k in range(args.seeds)])

    print(f"\n=== {partido['local']} vs {partido['visitante']} — Δ E[premio] de cubrir "
          f"cada marcador, por fila (base ${v_base:,.0f}, {args.seeds} semillas, "
          f"{n_sims} sorteos) ===")
    orden = np.argsort(-delta)
    for i in orden:
        k, nuevo, actual = etiquetas[i]
        num = numeros[k] if k < len(numeros) else f"fila{k}"
        marca = " ✅ supera el doble umbral" if (
            delta[i] > UMBRAL_ABS and delta[i] > UMBRAL_SE * se[i]) else ""
        print(f"  #{num} {actual} → {nuevo}: {delta[i]:+10,.0f} ± {se[i]:,.0f}{marca}")

    mejor = int(orden[0])
    if args.confirm_seeds <= 0 or not (
            delta[mejor] > UMBRAL_ABS and delta[mejor] > UMBRAL_SE * se[mejor]):
        print("\nVeredicto: NINGUNA cobertura supera el doble umbral del gate "
              f"(Δ > ${UMBRAL_ABS:,.0f} y > {UMBRAL_SE}·SE) — la planilla vigente "
              "se sostiene tal como está." if not (
                  delta[mejor] > UMBRAL_ABS and delta[mejor] > UMBRAL_SE * se[mejor])
              else "\n(confirmación desactivada: el Δ de arriba es best-of-N, "
                   "tomarlo con winner's curse)")
        return

    k, nuevo, actual = etiquetas[mejor]
    num = numeros[k] if k < len(numeros) else f"fila{k}"
    print(f"\nEl mejor ({actual} → {nuevo} en #{num}) supera el umbral en el barrido. "
          f"Confirmando en {args.confirm_seeds} semillas FRESCAS (best-of-"
          f"{len(variantes)} obliga)...")
    cdelta, cse, _ = medir(
        ev, base, [variantes[mejor]],
        [seed0 + CONFIRM_OFFSET + k for k in range(args.confirm_seeds)])
    ok = cdelta[0] > UMBRAL_ABS and cdelta[0] > UMBRAL_SE * cse[0]
    print(f"Confirmación: Δ = {cdelta[0]:+,.0f} ± {cse[0]:,.0f} → "
          + ("✅ CONFIRMADO: vale cambiar ese pick a mano al recargar."
             if ok else
             "❌ NO confirma: era winner's curse del barrido — no tocar la planilla."))


if __name__ == "__main__":
    main()

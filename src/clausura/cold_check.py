"""Control FRÍO del warm start: ¿la cadena de planillas quedó en un óptimo local?

Cada corrida hereda la planilla anterior como punto de partida del ascenso por
coordenadas (`picks.load_warm_start`). Eso vale +$3.622 ± 699 medido el 8/8 y baja
el churn de 49% a 22% — pero tiene un mecanismo de trinquete que nadie estaba
mirando (auditoría del 13/8):

  * un pick heredado solo se abandona si un candidato del **menú de hoy** lo supera;
    nunca se lo compara contra "¿entrarías al menú desde cero?", así que puede
    sobrevivir indefinidamente aunque hoy nadie lo elegiría;
  * la cadena lleva ~10 generaciones, cada una sembrada por la anterior;
  * y el gate del rerun compara **warm nueva vs warm vigente** — dos habitantes del
    mismo pozo. Ninguna corrida fría entra jamás a la comparación, así que si la
    cadena quedó atrapada, no hay ningún observable que lo diga.

Este módulo corre el pipeline en frío (mismos insumos, ascenso desde el ancla EV),
lo compara contra la planilla vigente con **sorteos comunes** —el mismo
`EvaluadorPortfolio` con semilla de evaluación independiente que usa el rerun— y
avisa solo si la fría gana de verdad: Δ > 2·SE y > $2.000, el mismo doble umbral
del gate por valor.

La corrida fría NO se versiona (`picks.run(guardar=False)`): guardarla la
convertiría en el warm start de la próxima corrida, que es justo la cadena que
viene a auditar. Cuando avisa, la decisión es humana — recargar desde una corrida
fría re-siembra la cadena.

Barato de correr: una corrida semanal en horario muerto. Si la fría nunca gana, el
trinquete queda documentado como teórico; si gana una vez, pagó todo el costo.

Uso:
    python -m src.clausura.cold_check                 # fecha actual, avisa si corresponde
    python -m src.clausura.cold_check --fecha 5
    python -m src.clausura.cold_check --dry-run       # sin Telegram ni estado
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "state" / "cold_check.json"

# Mismo doble umbral que el gate por valor del rerun (rerun_cierre.vale_avisar):
# el óptimo es plano y el churn produce Δ de cualquier signo, así que hace falta
# que sea distinguible del ruido Y que valga una recarga manual.
UMBRAL_SE = 2.0
UMBRAL_ABS = 2_000.0
EVAL_SEEDS = 5

# Historial que se guarda: sirve para ver si el trinquete aparece recién en fechas
# avanzadas (la cadena se hace más profunda con cada corrida).
MAX_HISTORIAL = 40


def matriz_vigente(contexto: dict, target_fecha: int, n_participaciones: int) -> np.ndarray | None:
    """Picks de la planilla VIGENTE sobre la grilla de la corrida fría.

    Se arma con `load_warm_start` —la misma función que alimenta la cadena, así que
    por construcción es exactamente lo que el ascenso habría heredado— y los huecos
    (-1: partidos que ninguna planilla previa cubre) se rellenan con la fría. Así las
    dos matrices difieren SOLO donde hay decisión heredada que auditar.
    """
    from src.clausura.picks import load_warm_start

    eventos = contexto["eventos"]
    warm = load_warm_start(eventos, target_fecha, n_participaciones)
    if warm is None:
        return None                     # no hay cadena todavía: nada que auditar
    fria = np.asarray(contexto["portfolio"].picks, dtype=np.int64)
    return np.where(warm >= 0, warm, fria)


def vale_avisar(comp, umbral_se: float = UMBRAL_SE, umbral_abs: float = UMBRAL_ABS) -> bool:
    """¿La corrida fría le gana a la cadena lo suficiente para actuar?"""
    return comp.delta > umbral_abs and comp.delta > umbral_se * comp.se


def formatear_alerta(comp, target_fecha: int, n_distintos: int, n_celdas: int) -> str:
    return (
        f"🧊 <b>La corrida FRÍA le gana a la cadena</b> — Fecha {target_fecha}\n\n"
        f"Δ E[premio] = <b>${comp.delta:,.0f}</b> ± {comp.se:,.0f} "
        f"(warm ${comp.valor_a:,.0f} → fría ${comp.valor_b:,.0f}, "
        f"{comp.n_seeds} semillas pareadas)\n"
        f"Picks distintos: {n_distintos}/{n_celdas}\n\n"
        f"El warm start viene heredando la planilla previa hace varias corridas y "
        f"parece haber quedado en un óptimo local: arrancar desde cero encuentra "
        f"algo mejor. Para re-sembrar la cadena, correr el pipeline normal con "
        f"<code>--cold</code> NO alcanza (no versiona): hay que decidir a mano si "
        f"recargar, y en ese caso re-generar la planilla y cargarla."
    )


def formatear_ok(comp, target_fecha: int) -> str:
    return (f"fecha {target_fecha}: la cadena aguanta — Δ fría−warm = "
            f"${comp.delta:,.0f} ± {comp.se:,.0f} "
            f"(warm ${comp.valor_a:,.0f} · fría ${comp.valor_b:,.0f})")


def guardar_historial(target_fecha: int, comp, avisó: bool, now: datetime) -> None:
    hist = []
    if STATE_PATH.exists():
        try:
            hist = json.loads(STATE_PATH.read_text(encoding="utf-8")).get("corridas", [])
        except Exception:                                      # noqa: BLE001
            hist = []
    hist.append({
        "ts": now.isoformat(),
        "fecha": target_fecha,
        "delta": float(comp.delta),
        "se": float(comp.se),
        "valor_warm": float(comp.valor_a),
        "valor_fria": float(comp.valor_b),
        "n_seeds": int(comp.n_seeds),
        "aviso": bool(avisó),
    })
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"corridas": hist[-MAX_HISTORIAL:]}, ensure_ascii=False, indent=1),
        encoding="utf-8")


def run(
    fecha: int | None = None,
    n_participaciones: int | None = None,
    n_sims: int | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> str | None:
    """Corre el control frío y devuelve el mensaje de alerta, o None si la cadena aguanta."""
    from src.clausura.picks import DEFAULT_SIMS, load_config, resolve_fecha
    from src.clausura.rivals import mis_numeros_env

    now = now or datetime.now(timezone.utc)
    if fecha is None:
        fecha = resolve_fecha("auto")
    if n_participaciones is None:
        n_participaciones = len(mis_numeros_env()) or 5
    if n_sims is None:
        n_sims = DEFAULT_SIMS
    load_config()          # falla temprano y claro si el config no está

    from src.clausura.picks import run as picks_run

    contexto: dict = {}
    log.info("corrida FRÍA de control: fecha %d, %d participaciones, %d sorteos",
             fecha, n_participaciones, n_sims)
    picks_run(fecha, n_participaciones, telegram=False, n_sims=n_sims,
              contexto=contexto, usar_warm_start=False, guardar=False)

    ev = contexto.get("evaluador")
    if ev is None or contexto.get("portfolio") is None:
        log.error("la corrida fría no dejó evaluador en el contexto — sin comparación")
        return None

    warm = matriz_vigente(contexto, fecha, n_participaciones)
    if warm is None:
        log.info("no hay planilla previa: la cadena todavía no existe, nada que auditar")
        return None

    fria = np.asarray(contexto["portfolio"].picks, dtype=np.int64)
    comp = ev.comparar(warm, fria, n_seeds=EVAL_SEEDS)
    n_distintos = int((warm != fria).sum())
    log.info("fría vs cadena: %s · %d/%d celdas distintas",
             comp, n_distintos, warm.size)

    avisar = vale_avisar(comp)
    if not dry_run:
        guardar_historial(fecha, comp, avisar, now)

    if not avisar:
        print(formatear_ok(comp, fecha))
        return None

    msg = formatear_alerta(comp, fecha, n_distintos, warm.size)
    print(msg.replace("<b>", "").replace("</b>", "")
             .replace("<code>", "").replace("</code>", ""))
    if not dry_run:
        from src.notifier.telegram import TelegramConfig, TelegramNotifier
        TelegramNotifier(TelegramConfig.from_env()).send(msg)
    return msg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", type=int, default=None,
                    help="default: la fecha en curso")
    ap.add_argument("--participaciones", type=int, default=None)
    ap.add_argument("--sims", type=int, default=None,
                    help="default: los mismos sorteos que producción")
    ap.add_argument("--dry-run", action="store_true",
                    help="comparar e imprimir sin Telegram ni historial")
    args = ap.parse_args()
    run(fecha=args.fecha, n_participaciones=args.participaciones,
        n_sims=args.sims, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

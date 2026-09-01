"""Gate por valor de la corrida diaria contra LO CARGADO en la web.

EL HUECO QUE CIERRA. `rerun_cierre` avisa solo si un cambio vale plata, pero
compara contra *la planilla vigente* — que a la tarde ya es la salida de la
corrida de las 11:00, la cual reescribe todo sin medir nada. Un cambio que
introduce la propia corrida diaria nunca se evaluaba por valor ni se señalaba
contra lo que el usuario ya tenía cargado. En la Fecha 4 eso costó puntos
reales: el domingo 30/8 a las 11:19 la corrida movió dos filas al 0-1 del
clásico ⭐x2 (el exacto), el usuario había cargado 1-1 el día anterior, y nadie
le dijo "esto cambió y vale $X" — 2 pts en vez de 16 en cada fila.

LA REFERENCIA. "Lo cargado" no se puede leer del API antes del cierre (gate por
partido), pero las marcas del modo carga (src.clausura.carga_state) guardan EL
VALOR que el usuario confirmó al cargar cada celda: clave
`carga:v2:{fecha}:{col}:{evento}` → "gl-gv". Es la única memoria pre-cierre de
lo que hay en la web, y ya existe — este módulo solo la lee.

LA REGLA. Después de optimizar, si la planilla nueva difiere de lo marcado como
cargado en partidos AÚN ABIERTOS de la fecha objetivo, se liquida el par
(cargado vs nuevo) con sorteos comunes — el mismo EvaluadorPortfolio y los
mismos umbrales del rerun (UMBRAL_SE, UMBRAL_ABS: recargar a mano tiene costo
real de tiempo y de tipeo):

  * si el Δ NO paga → la planilla ADOPTA lo cargado en esas celdas. El churn del
    optimizador es ruido conocido (43/96 picks con insumos idénticos) y no
    amerita ni recarga ni drift silencioso: planilla y web quedan iguales.
  * si el Δ paga (o no se pudo medir) → los picks nuevos quedan y el aviso lista
    exactamente qué celdas recargar y cuánto vale.

Una marca puede estar vieja (el usuario cambió la web sin re-marcar): sigue
siendo la mejor referencia disponible, y el drift_audit post-cierre adopta la
web como verdad igual que siempre.

Apagado de emergencia: CLAUSURA_GATE_CARGA=0.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

import numpy as np

from src.clausura.economics import score_index
from src.clausura.scoring import expected_points_grid

log = logging.getLogger(__name__)

MARCA_RE = re.compile(r"^carga:v2:(\d{1,3}):(\d{1,3}):(\d{1,12})$")
VALOR_RE = re.compile(r"^(\d{1,2})-(\d{1,2})$")


def marcas_de_fecha(fecha_n: int, marcas: dict[str, str] | None = None
                    ) -> dict[tuple[int, int], tuple[int, int]]:
    """(col, evento_id) → (gl, gv) de las marcas del modo carga para una fecha.

    `marcas` inyectable para tests; default: el estado compartido del servidor.
    Claves de especiales (`carga:v2:esp:*`) y valores malformados se ignoran —
    las escribe el navegador y acá solo se confía en lo que parsea limpio.
    """
    if marcas is None:
        from src.clausura.carga_state import leer
        marcas = leer()
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for k, v in marcas.items():
        mk = MARCA_RE.match(k)
        if not mk or int(mk.group(1)) != fecha_n:
            continue
        mv = VALOR_RE.match(str(v).strip())
        if not mv:
            continue
        out[(int(mk.group(2)), int(mk.group(3)))] = (int(mv.group(1)), int(mv.group(2)))
    return out


def diffs_vs_cargado(
    payload: dict,
    cargado: dict[tuple[int, int], tuple[int, int]],
    now: datetime,
    n_participaciones: int,
) -> list[tuple[dict, list[tuple[int, tuple[int, int], tuple[int, int]]]]]:
    """Celdas de partidos AÚN ABIERTOS donde la planilla nueva ≠ lo cargado.

    [(fila_de_payload, [(col, cargado, nuevo)])]. Celdas sin marca no cuentan:
    si el usuario no cargó, el pick nuevo no le pisa nada.
    """
    out = []
    for row in payload.get("picks", []):
        eid = int(row["evento_id"])
        if datetime.fromisoformat(row["cierre_pronostico_utc"]) <= now:
            continue
        cambios = []
        for col, sc in enumerate(row.get("scores", [])[:n_participaciones]):
            marca = cargado.get((col, eid))
            if marca is not None and tuple(marca) != (int(sc[0]), int(sc[1])):
                cambios.append((col, marca, (int(sc[0]), int(sc[1]))))
        if cambios:
            out.append((row, cambios))
    return out


def _adoptar_cargado(payload, port, diffs, idx_of, grids) -> int:
    """Vuelve las celdas en `diffs` a lo cargado, en la matriz del portfolio Y en
    el payload (scores, e_pts, picks_temporada) — la planilla que se guarda es el
    warm start de mañana y el prev del rerun de la tarde: si solo se tocara una
    de las dos copias, el desfasaje renacería en la próxima corrida."""
    temporada_by_eid = {int(r["evento_id"]): r
                       for r in payload.get("picks_temporada", [])}
    n = 0
    for row, cambios in diffs:
        eid = int(row["evento_id"])
        col_ev = idx_of[eid]
        pref = bool(row.get("preferencial"))
        for k, marca, _nuevo in cambios:
            port.picks[k, col_ev] = score_index(*marca)
            row["scores"][k] = list(marca)
            row["e_pts"][k] = round(
                expected_points_grid(marca, grids[col_ev], pref), 2)
            t = temporada_by_eid.get(eid)
            if t is not None:
                t["scores"][k] = list(marca)
            n += 1
    return n


def aplicar_gate(
    payload: dict,
    port,
    idx_of: dict[int, int],
    grids,
    target_fecha: int,
    n_participaciones: int,
    mis_numeros: list[int],
    now: datetime | None = None,
    marcas: dict[str, str] | None = None,
) -> str | None:
    """Corre el gate y devuelve el texto para Telegram (None si no hay nada que
    decir). Muta `payload` y `port.picks` cuando adopta lo cargado.
    """
    if os.environ.get("CLAUSURA_GATE_CARGA", "1") == "0":
        log.info("gate por carga APAGADO por CLAUSURA_GATE_CARGA=0")
        return None
    now = now or datetime.now(timezone.utc)
    cargado = marcas_de_fecha(target_fecha, marcas)
    if not cargado:
        log.info("sin marcas de carga de la fecha %d — gate por carga no aplica "
                 "(nada cargado que proteger)", target_fecha)
        return None
    diffs = diffs_vs_cargado(payload, cargado, now, n_participaciones)
    if not diffs:
        log.info("la planilla nueva coincide con lo cargado (%d marcas) — sin drift",
                 len(cargado))
        return None
    n_picks = sum(len(c) for _, c in diffs)

    comp = None
    ev = getattr(port, "evaluador", None)
    if ev is not None:
        try:
            import gc
            gc.collect()
            matriz_cargada = np.array(port.picks, dtype=np.int64).copy()
            for row, cambios in diffs:
                col_ev = idx_of[int(row["evento_id"])]
                for k, marca, _nuevo in cambios:
                    matriz_cargada[k, col_ev] = score_index(*marca)
            from src.clausura.rerun_cierre import EVAL_SEEDS
            comp = ev.comparar(matriz_cargada, port.picks, n_seeds=EVAL_SEEDS)
            log.info("gate por carga — valor del cambio vs lo cargado: %s", comp)
        except Exception as e:                                # noqa: BLE001
            log.error("gate por carga: no pude medir el Δ (%s) — aviso igual; el "
                      "gate está inhabilitado mientras esto falle", e, exc_info=True)

    from src.clausura.rerun_cierre import vale_avisar
    avisar = comp is None or vale_avisar(comp)

    payload["veredicto_carga"] = {
        "avisar": bool(avisar),
        "medido": comp is not None,
        "n_picks": int(n_picks),
        **({} if comp is None else {
            "delta": float(comp.delta), "se": float(comp.se),
            "valor_a": float(comp.valor_a), "valor_b": float(comp.valor_b),
            "n_seeds": int(comp.n_seeds), "significativa": bool(comp.significativa),
        }),
    }

    if not avisar:
        n = _adoptar_cargado(payload, port, diffs, idx_of, grids)
        log.info("gate por carga: %d pick(s) vuelven a lo YA CARGADO (%s) — el "
                 "cambio del optimizador no paga la recarga", n, comp)
        return (f"🧷 <i>{n} pick(s) que ya cargaste se respetan tal cual: el cambio "
                f"que proponía el optimizador vale {comp.delta:+,.0f} ± {comp.se:,.0f}"
                f" y no paga una recarga.</i>")

    lines = ["<b>⚠️ Esta corrida cambió picks que YA CARGASTE en la web</b>"]
    if comp is not None:
        lines.append(f"Recargarlos vale <b>{comp.delta:+,.0f}</b> ± {comp.se:,.0f} "
                     f"de E[premio] (${comp.valor_a:,.0f} → ${comp.valor_b:,.0f}).")
    else:
        lines.append("No pude medir cuánto vale (evaluador caído) — aviso por las dudas.")
    for row, cambios in diffs:
        pref = " ⭐x2" if row.get("preferencial") else ""
        cierre_iso = row.get("cierre_pronostico_utc", "")
        lines.append(f"\n<b>{row['partido']}</b>{pref}")
        for k, marca, nuevo in cambios:
            numero = mis_numeros[k] if k < len(mis_numeros) else f"col{k + 1}"
            lines.append(f"  {numero}: cargaste {marca[0]}-{marca[1]} → "
                         f"ahora dice <b>{nuevo[0]}-{nuevo[1]}</b>")
    lines.append("\nActualizá SOLO estos picks en la web y tocá la fila en el modo "
                 "carga para re-marcarla.")
    if comp is not None and not comp.significativa:
        lines.append("⚠️ El Δ no supera su ruido de simulación: el aviso sale por el "
                     "piso absoluto, no porque la mejora esté confirmada.")
    return "\n".join(lines)

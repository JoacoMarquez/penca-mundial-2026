"""Postmortem automático por fecha (regla de trabajo #4 del proyecto).

Cuando una fecha tiene suficientes partidos jugados (MIN_JUGADOS, no todos: un
suspendido no puede bloquear el análisis de los demás), genera y manda por Telegram
un análisis de qué pasó, y persiste el detalle en JSON:

  - Resultados reales + puntos de nuestras 12 participaciones por partido
    (kernel real de Supermatch, estrella x2), esperado vs real — el E[pts] se toma
    de la planilla generada ANTES del cierre de cada partido, porque en las
    posteriores la grilla es una delta y el "esperado" sería el puntaje real.
  - Distribución de puntos del POOL en la fecha (desde el snapshot de picks
    públicos): mediana, máximo, cuántos rivales nos ganaron, y si alguna nuestra
    ganó el premio por fecha.
  - Qué pegó y qué no: exactos nuestros vs exactos del pool por partido, y el
    rendimiento en el partido estrella (x2).
  - Chequeo de ASIGNACIÓN (el objetivo real: que UNA fila se despegue, no acertar
    en promedio): P(N rivales al azar tengan un máximo ≥ el nuestro), filas en el
    top 10% del pool, nivel (promedio vs pool) y concentración (corr/picks iguales
    entre filas). Por fecha y acumulado de temporada (ranking de la web).
  - Insumos de recalibración: tasa de exactos del pool observada en la fecha
    (la calibración en sí ya es online: el pipeline de picks la lee del ranking, y
    la Q empírica del pool sale del snapshot — acá se REPORTA, no se duplica).

Idempotente: un archivo por fecha en data/postmortems/clausura/fecha_NN.json.
El timer corre de madrugada; si la última fecha completa ya tiene postmortem,
sale en silencio.

Uso:
    python -m src.clausura.postmortem                # detecta la fecha, genera y manda
    python -m src.clausura.postmortem --fecha 3      # fuerza una fecha puntual
    python -m src.clausura.postmortem --dry-run      # imprime sin Telegram ni estado
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.clausura.api import PencaApiClient, resultado_finalizado
from src.clausura.rivals import mis_numeros_env
from src.clausura.scoring import supermatch_points

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PM_DIR = ROOT / "data" / "postmortems" / "clausura"


@dataclass
class FechaStats:
    fecha_n: int
    resultados: dict[int, tuple[int, int]]              # evento_id → real
    eventos: list[dict]                                  # los de la fecha, con metadata
    # nuestras participaciones
    puntos: dict[int, int] = field(default_factory=dict)          # numero → pts fecha
    exactos: dict[int, int] = field(default_factory=dict)         # numero → exactos fecha
    esperados: dict[int, float] = field(default_factory=dict)     # numero → ΣE[pts] (si hay)
    detalle: dict[int, dict[int, dict]] = field(default_factory=dict)  # eid → numero → {...}
    # pool
    pool_puntos: list[int] = field(default_factory=list)
    pool_exactos_por_evento: dict[int, float] = field(default_factory=dict)  # eid → frac


# -------------------- datos --------------------

def resultados_de_fecha(
    cfg: dict, fecha_n: int, min_jugados: int = 1,
) -> tuple[dict[int, tuple[int, int]], int] | None:
    """(evento_id → (gl, gv) de los JUGADOS, cuántos faltan). None si no alcanza.

    Antes exigía la fecha COMPLETA y devolvía None si faltaba un solo resultado. Con
    eso el postmortem —el único circuito que compara lo pronosticado contra lo que
    pasó— nunca corrió: la Fecha 1 tiene Torque–Peñarol SUSPENDIDO, y un suspendido
    no se resuelve en días. Un partido que se reprograma para dentro de un mes
    bloqueaba el aprendizaje de los otros siete.

    Ahora analiza lo jugado y REPORTA lo que falta. El postmortem es idempotente por
    fecha, así que si después se juega el pendiente se puede regenerar y el nuevo
    archivo incluye todo.
    """
    nombre = f"Fecha {fecha_n}"
    f = cfg["fechas"].get(nombre)
    if f is None:
        raise ValueError(f"no existe {nombre} en el config")
    out: dict[int, tuple[int, int]] = {}
    faltan = 0
    with PencaApiClient() as api:
        data = api._get(f"/front/campeonatos/fechas/{f['fecha_id']}/eventos")
    for e in data:
        real = resultado_finalizado(e)
        if real is None:
            faltan += 1
            continue
        out[int(e["id"])] = real
    if len(out) < min_jugados:
        return None
    return out, faltan


def picks_de_planilla(fecha_n: int, mis_numeros: list[int]) -> tuple[dict, dict]:
    """(numero → {evento_id → pick}, evento_id → {numero → e_pts PRE-PARTIDO}).

    Los PICKS salen de la última planilla (la que refleja lo cargado). El `e_pts`,
    en cambio, sale de la última versión generada ANTES del cierre de cada partido.

    La distinción no es cosmética, es lo que hace que "esperado vs real" signifique
    algo. Para un partido ya jugado `build_season_grids` usa una grilla DELTA —toda
    la masa en el resultado— así que el `e_pts` que guarda una planilla posterior ES
    el puntaje obtenido. Comparar eso contra lo real daba identidad perfecta en las
    cinco filas y no medía nada (detectado el 2026-08-10).
    """
    from src.clausura.picks import fecha_dir
    from src.utils.versions import latest_version, version_num

    d = fecha_dir(fecha_n)
    latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
    if latest is None:
        raise FileNotFoundError(f"no hay planilla guardada para la fecha {fecha_n}")
    data = json.loads(latest.read_text(encoding="utf-8"))
    picks: dict[int, dict[int, tuple[int, int]]] = {n: {} for n in mis_numeros}
    for row in data.get("picks", []):
        eid = int(row["evento_id"])
        for k, score in enumerate(row.get("scores", [])):
            if k < len(mis_numeros):
                picks[mis_numeros[k]][eid] = (int(score[0]), int(score[1]))

    # e_pts pre-partido: recorriendo las versiones de la MÁS VIEJA a la más nueva y
    # quedándose con la última cuyo `generado_utc` sea anterior al cierre del evento.
    e_pts: dict[int, dict[int, float]] = {}
    for path in sorted(d.glob("v*_*.json"), key=version_num):
        payload = json.loads(path.read_text(encoding="utf-8"))
        generado = payload.get("generado_utc")
        if not generado:
            continue
        gen = datetime.fromisoformat(generado)
        for row in payload.get("picks", []):
            if "e_pts" not in row or not row.get("cierre_pronostico_utc"):
                continue
            if gen >= datetime.fromisoformat(row["cierre_pronostico_utc"]):
                continue                    # generada con el partido ya cerrado
            e_pts[int(row["evento_id"])] = {
                mis_numeros[k]: float(v)
                for k, v in enumerate(row["e_pts"]) if k < len(mis_numeros)
            }
    return picks, e_pts


def latest_snapshot_participaciones() -> list[dict]:
    """Participaciones del último snapshot FRESCO (≤48h, la misma cota que picks).

    Sin la cota: si el escaneo del ExecStartPre aborta (429), el postmortem
    comparaba contra un snapshot sin los picks de los partidos recién cerrados →
    puntos del pool subestimados → "ganaste la fecha" en el reporte sin haberla
    ganado. Mejor un postmortem sin distribución del pool que uno que miente.
    """
    from src.clausura.pool_snapshot import load_latest_snapshot
    snap = load_latest_snapshot(max_age_hours=48)
    if snap is None:
        log.warning("sin snapshot fresco (≤48h) — el postmortem sale sin "
                    "distribución del pool en vez de usar una vieja que miente")
        return []
    return snap.get("participaciones", [])


# -------------------- cómputo (puro, testeable) --------------------

def compute_stats(
    fecha_n: int,
    eventos_fecha: list[dict],
    resultados: dict[int, tuple[int, int]],
    picks: dict[int, dict[int, tuple[int, int]]],
    e_pts: dict[int, dict[int, float]],
    pool_participaciones: list[dict],
    mis_numeros: set[int],
) -> FechaStats:
    st = FechaStats(fecha_n=fecha_n, resultados=resultados, eventos=eventos_fecha)
    pref_de = {ev["evento_id"]: bool(ev.get("preferencial")) for ev in eventos_fecha}

    for numero, mios in picks.items():
        total = exactos = 0
        esperado = 0.0
        tiene_esperado = False
        for eid, real in resultados.items():
            pick = mios.get(eid)
            if pick is None:
                continue
            pts = supermatch_points(pick, real, pref_de.get(eid, False))
            total += pts
            if pick == real:
                exactos += 1
            st.detalle.setdefault(eid, {})[numero] = {"pick": pick, "pts": pts}
            if eid in e_pts and numero in e_pts[eid]:
                esperado += e_pts[eid][numero]
                tiene_esperado = True
        st.puntos[numero] = total
        st.exactos[numero] = exactos
        if tiene_esperado:
            st.esperados[numero] = round(esperado, 1)

    # pool: puntos de la fecha por rival (desde el snapshot de picks públicos)
    exact_hits: dict[int, int] = {eid: 0 for eid in resultados}
    n_con_picks = 0
    for r in pool_participaciones:
        if int(r.get("numero", -1)) in mis_numeros:
            continue
        rp = {int(k): (int(v[0]), int(v[1])) for k, v in r.get("picks", {}).items()}
        relevantes = {eid: rp[eid] for eid in resultados if eid in rp}
        if not relevantes:
            continue
        n_con_picks += 1
        pts = 0
        for eid, pick in relevantes.items():
            pts += supermatch_points(pick, resultados[eid], pref_de.get(eid, False))
            if pick == resultados[eid]:
                exact_hits[eid] += 1
        st.pool_puntos.append(pts)
    if n_con_picks:
        st.pool_exactos_por_evento = {eid: h / n_con_picks for eid, h in exact_hits.items()}
    return st


# -------------------- chequeo de asignación (portfolio, no acierto) --------------------
#
# El objetivo del sistema no es acertar en promedio sino ASIGNAR resultados a las
# participaciones para que UNA se despegue. El postmortem clásico (puntos por fila,
# percentil) no lo mide: doce filas en el percentil 78 son un desastre para el
# objetivo aunque cada una "esté bien". Detectado el 2026-08-15 con 11 partidos:
# la diversificación mecánica era correcta (corr 0.08 entre filas) pero 13 tickets
# del pool elegidos AL AZAR le ganaban a nuestro máximo el 97% de las veces — el
# portfolio pagaba el costo de diferenciarse sin cobrar la prima.
#
# La vara es esa: si N rivales cualesquiera producen un máximo mejor que el nuestro
# casi siempre, la asignación no está funcionando, sea por nivel (filas todas bajo la
# mediana) o por concentración (filas que se pisan). Se reportan las dos causas.

def _log_comb(n: int, k: int) -> float:
    from math import lgamma
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def prob_azar_gana(pool: list[int], nuestro_max: int, n_tickets: int) -> float | None:
    """P(el máximo de `n_tickets` rivales tomados al azar SIN reposición ≥ nuestro máx).

    Exacta (hipergeométrica), sin sorteos: P(todos < m) = C(k, N)/C(M, N) con k el
    número de rivales estrictamente por debajo de m. None si el pool no alcanza.
    """
    m = len(pool)
    if n_tickets <= 0 or m < n_tickets:
        return None
    k = sum(1 for p in pool if p < nuestro_max)
    if k < n_tickets:
        return 1.0
    p_todos_abajo = np.exp(_log_comb(k, n_tickets) - _log_comb(m, n_tickets))
    return float(1.0 - p_todos_abajo)


def chequeo_asignacion(st: FechaStats) -> dict | None:
    """Métricas de portfolio de la fecha. None sin pool o sin filas nuestras.

    - `p_azar_gana`: la vara — P(máx de N tickets al azar del pool ≥ nuestro máx).
    - `filas_top10`: cuántas nuestras entran en el decil superior del pool.
    - `promedio_nuestro` vs `promedio_pool`: causa "nivel".
    - `corr_filas`: correlación media de puntos partido a partido entre nuestras
      filas (0 ≈ independientes); `coincidencia_pares` es la fracción de picks
      iguales entre pares de filas nuestras: causa "concentración".
    """
    if not st.pool_puntos or not st.puntos:
        return None
    pool = np.array(st.pool_puntos)
    nuestros = np.array(list(st.puntos.values()))
    n = len(nuestros)
    nuestro_max = int(nuestros.max())
    p90 = float(np.percentile(pool, 90))
    out = {
        "n_filas": n,
        "nuestro_max": nuestro_max,
        "p_azar_gana": prob_azar_gana(st.pool_puntos, nuestro_max, n),
        "filas_top10": int((nuestros >= p90).sum()),
        "p90_pool": round(p90, 1),
        "promedio_nuestro": round(float(nuestros.mean()), 2),
        "promedio_pool": round(float(pool.mean()), 2),
        "mediana_pool": float(np.median(pool)),
        "filas_bajo_mediana": int((nuestros < np.median(pool)).sum()),
    }
    # concentración: matriz filas × partidos de puntos y de picks
    numeros = list(st.puntos)
    eids = [eid for eid in st.resultados if eid in st.detalle]
    if n >= 2 and len(eids) >= 2:
        M = np.array([[st.detalle[eid].get(num, {}).get("pts", 0) for eid in eids]
                      for num in numeros], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(M)
        C = np.nan_to_num(C, nan=0.0)          # fila constante ⇒ corr indefinida ⇒ 0
        out["corr_filas"] = round(float((C.sum() - n) / (n * n - n)), 3)
        iguales = total = 0
        for i in range(n):
            for j in range(i + 1, n):
                for eid in eids:
                    a = st.detalle[eid].get(numeros[i], {}).get("pick")
                    b = st.detalle[eid].get(numeros[j], {}).get("pick")
                    if a is None or b is None:
                        continue
                    total += 1
                    iguales += a == b
        out["coincidencia_pares"] = round(iguales / total, 3) if total else None
    return out


def chequeo_asignacion_temporada(
    ranking_puntos: dict[int, int], mis_numeros: set[int],
) -> dict | None:
    """Mismo chequeo sobre los puntos ACUMULADOS del ranking de la web.

    Es el que importa: el premio grande es al final. Usa `puntos_totales` del
    ranking (la web es la verdad; incluye lo que la web ya liquidó y nada más), así
    no depende de tener la serie de postmortems previos completa.
    """
    nuestros = [p for num, p in ranking_puntos.items() if num in mis_numeros]
    pool = [p for num, p in ranking_puntos.items() if num not in mis_numeros]
    if not nuestros or not pool:
        return None
    arr = np.array(pool)
    nuestro_max = max(nuestros)
    p90 = float(np.percentile(arr, 90))
    return {
        "n_filas": len(nuestros),
        "nuestro_max": nuestro_max,
        "puesto_max": int((arr > nuestro_max).sum()) + 1,
        "p_azar_gana": prob_azar_gana(pool, nuestro_max, len(nuestros)),
        "filas_top10": int(sum(1 for p in nuestros if p >= p90)),
        "p90_pool": round(p90, 1),
        "promedio_nuestro": round(float(np.mean(nuestros)), 2),
        "promedio_pool": round(float(arr.mean()), 2),
        "mediana_pool": float(np.median(arr)),
        "filas_bajo_mediana": int(sum(1 for p in nuestros if p < np.median(arr))),
        "max_pool": int(arr.max()),
    }


def formatear_asignacion(fecha: dict | None, temporada: dict | None) -> str:
    """Sección del reporte. Vacía si no hay nada que decir."""
    if not fecha and not temporada:
        return ""
    lines = ["<b>🎯 Asignación</b> (¿alguna fila se despega?)"]

    def bloque(nombre: str, d: dict) -> None:
        p = d.get("p_azar_gana")
        p_txt = f"{p:.0%}" if p is not None else "n/d"
        alerta = " ⚠️" if p is not None and p >= 0.8 else ""
        puesto = f" (#{d['puesto_max']})" if "puesto_max" in d else ""
        lines.append(
            f"  {nombre}: máx nuestro <b>{d['nuestro_max']}</b>{puesto} · "
            f"{d['n_filas']} rivales al azar nos ganan el <b>{p_txt}</b>{alerta} · "
            f"{d['filas_top10']}/{d['n_filas']} en top 10% (≥{d['p90_pool']:.0f})")
        causa = (f"    nivel: promedio {d['promedio_nuestro']:.1f} vs pool "
                 f"{d['promedio_pool']:.1f}, {d['filas_bajo_mediana']}/{d['n_filas']} "
                 f"bajo la mediana ({d['mediana_pool']:.0f})")
        if d.get("corr_filas") is not None:
            causa += (f" · concentración: corr {d['corr_filas']:.2f}, "
                      f"picks iguales entre filas {d['coincidencia_pares']:.0%}")
        lines.append(causa)

    if fecha:
        bloque("Fecha", fecha)
    if temporada:
        bloque("Temporada", temporada)
    lines.append("  <i>Vara: si N rivales al azar le ganan a nuestro máx casi siempre, "
                 "el portfolio paga la diferenciación sin cobrarla — mirar el PIT.</i>")
    return "\n".join(lines)


def comparar_puntos_publicados(
    calculados: dict[int, int], publicados: dict[int, int],
) -> list[str]:
    """Diferencias entre nuestros puntos calculados y los del ranking de la web.

    Los RIVALES ya están anclados a la verdad (el residuo del RivalModel corrige
    cada fila contra el ranking), pero nuestro lado del simulador sale del kernel
    propio y nunca se contrastaba. Cualquier divergencia de liquidación —cambio de
    reglas mid-torneo (el Mundial lo tuvo, con recálculo retroactivo), un
    suspendido liquidado raro— sesgaba SOLO nuestra posición simulada, sin síntoma.
    """
    difs = []
    for numero in sorted(calculados):
        pub = publicados.get(numero)
        if pub is not None and pub != calculados[numero]:
            difs.append(f"  {numero % 1000:03d}: calculado {calculados[numero]} vs web {pub}")
    return difs


def _totales_calculados(fecha: int, puntos_fecha: dict[int, int]) -> dict[int, int] | None:
    """Puntos de temporada por participación: postmortems previos + la fecha actual.

    None si falta el postmortem de alguna fecha anterior — sin la serie completa
    la comparación contra `puntos_totales` del ranking daría falsas alarmas.
    """
    totales = dict(puntos_fecha)
    for n in range(1, fecha):
        p = pm_path(n)
        if not p.exists():
            return None
        prev = json.loads(p.read_text(encoding="utf-8")).get("puntos") or {}
        for numero_str, pts in prev.items():
            totales[int(numero_str)] = totales.get(int(numero_str), 0) + int(pts)
    return totales


def _percentil(valor: int, pool: list[int]) -> float:
    """Fracción del pool que quedó ESTRICTAMENTE por debajo."""
    if not pool:
        return 0.0
    return sum(1 for p in pool if p < valor) / len(pool)


# -------------------- reporte --------------------

def formatear_postmortem(
    st: FechaStats, premio_fecha: float | None = None, faltan: int = 0,
) -> str:
    ev_by_id = {ev["evento_id"]: ev for ev in st.eventos}
    lines = [f"<b>📊 Postmortem — Fecha {st.fecha_n}</b>"]

    # partidos: resultado + cómo nos fue + exactos del pool
    lines.append("")
    for eid, real in st.resultados.items():
        ev = ev_by_id.get(eid, {})
        pref = " ⭐" if ev.get("preferencial") else ""
        det = st.detalle.get(eid, {})
        nuestros = [d["pts"] for d in det.values()]
        exactos_nuestros = sum(1 for d in det.values() if d["pick"] == real)
        pool_ex = st.pool_exactos_por_evento.get(eid)
        pool_txt = f" · pool exacto {pool_ex:.0%}" if pool_ex is not None else ""
        mejor = max(nuestros) if nuestros else 0
        lines.append(
            f"{ev.get('local', '?')} <b>{real[0]}-{real[1]}</b> {ev.get('visitante', '?')}{pref}"
            f" — nuestras: mejor {mejor} pts, {exactos_nuestros}/{max(len(det), 1)} "
            f"exacto{pool_txt}")

    # participaciones: real (vs esperado), percentil en el pool
    lines.append("")
    lines.append("<b>Por participación</b> (pts fecha · vs pool)")
    for numero in sorted(st.puntos, key=lambda n: -st.puntos[n]):
        pts = st.puntos[numero]
        pct = _percentil(pts, st.pool_puntos)
        esp = st.esperados.get(numero)
        esp_txt = f" (E {esp:.0f})" if esp is not None else ""
        ex = st.exactos[numero]
        ex_txt = f" · {ex} exacto{'s' if ex != 1 else ''}" if ex else ""
        lines.append(f"  {numero % 1000:03d}: <b>{pts}</b>{esp_txt} · top {1 - pct:.0%}{ex_txt}")

    # pool
    if st.pool_puntos:
        arr = np.array(st.pool_puntos)
        mejor_nuestra = max(st.puntos.values(), default=0)
        nos_ganaron = int((arr > mejor_nuestra).sum())
        lines.append("")
        lines.append(f"<b>Pool</b>: {len(arr)} rivales · mediana {int(np.median(arr))} · "
                     f"máx {int(arr.max())} · {nos_ganaron} arriba de nuestra mejor")
        if premio_fecha and nos_ganaron == 0 and mejor_nuestra >= int(arr.max()):
            # El 🏆 solo con la fecha LIQUIDADA: con un suspendido pendiente el
            # premio no está definido (Arts. 9 y 14) — la Fecha 1 tuvo exactamente
            # este estado (Torque–Peñarol suspendido) y el reporte cantaba victoria.
            if faltan == 0:
                lines.append(f"🏆 Nuestra mejor participación empató o ganó la fecha "
                             f"(premio ${premio_fecha:,.0f})")
            else:
                lines.append(f"⏳ Vamos ganando la fecha, pero falta{'n' if faltan != 1 else ''} "
                             f"{faltan} partido{'s' if faltan != 1 else ''} — nada liquidado todavía")
        pool_exact_rate = float(np.mean(list(st.pool_exactos_por_evento.values()))) \
            if st.pool_exactos_por_evento else None
        if pool_exact_rate is not None:
            lines.append(f"Tasa de exactos del pool en la fecha: {pool_exact_rate:.1%} "
                         f"(insumo de calibración — el pipeline la absorbe solo)")
    else:
        lines.append("")
        lines.append("<i>Sin snapshot del pool para esta fecha — comparación vs rivales omitida.</i>")

    return "\n".join(lines)


# -------------------- persistencia / orquestación --------------------

def pm_path(fecha_n: int) -> Path:
    return PM_DIR / f"fecha_{fecha_n:02d}.json"


# Cuántos partidos de la fecha tienen que estar jugados para que valga un postmortem.
# 6 de 8 deja afuera la fecha en curso (viernes/sábado) pero no se traba con uno o dos
# suspendidos, que es lo que impedía que este módulo corriera alguna vez.
MIN_JUGADOS = 6


def _resultados_guardados(n: int) -> dict[int, tuple[int, int]]:
    """Resultados que tiene el postmortem ya escrito de esa fecha. Vacío si no hay.

    Devuelve los VALORES y no la cantidad: comparar solo conteos dejaba pasar en
    silencio una corrección del proveedor (mismo número de resultados, un marcador
    distinto) — el postmortem quedaba escrito con el resultado viejo para siempre.
    """
    p = pm_path(n)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8")).get("resultados") or {}
        return {int(k): (int(v[0]), int(v[1])) for k, v in raw.items()}
    except Exception:                                          # noqa: BLE001
        return {}


def fecha_a_analizar(cfg: dict) -> int | None:
    """La fecha más vieja cuyo postmortem falta o quedó INCOMPLETO.

    "Suficientes" y no "todos": un partido suspendido —Torque–Peñarol en la Fecha 1—
    se reprograma para dentro de semanas, y exigir la fecha completa dejaba el
    análisis de los otros siete sin correr nunca.

    Pero disparar con el mínimo y no volver nunca congela datos parciales: el
    postmortem de la Fecha 1 se escribió el 2026-08-10 con **6 de 8** partidos —le
    faltaban Torque–Peñarol y Maldonado–Racing, que se jugaron esa misma noche— y no
    se iba a rehacer jamás. La herramienta que existe para aprender de lo que pasó
    quedaba permanentemente equivocada sobre la primera fecha.

    Por eso la condición no es "existe el archivo" sino "el archivo tiene TODOS los
    resultados que hoy se pueden ver". Se rehace solo cuando aparecen más.
    """
    nums = sorted(int(n.split()[-1]) for n in cfg["fechas"])
    for n in nums:
        res = resultados_de_fecha(cfg, n, min_jugados=MIN_JUGADOS)
        if res is None:
            break   # las fechas van en orden: si esta no llegó, las siguientes tampoco
        disponibles, _ = res
        guardados = _resultados_guardados(n)
        # Regenerar solo ante resultados NUEVOS o CAMBIADOS. Un resultado que
        # desaparece del API (glitch transitorio) no pisa un postmortem bueno.
        if not any(guardados.get(eid) != real for eid, real in disponibles.items()):
            continue          # al día: mismos resultados, mismos valores
        if guardados:
            log.info("rehaciendo el postmortem de la fecha %d: tenía %d resultados y "
                     "ahora hay %d (o cambió algún marcador)",
                     n, len(guardados), len(disponibles))
        return n    # una por corrida: la más vieja pendiente
    return None


def run(fecha: int | None = None, dry_run: bool = False) -> str | None:
    from src.clausura.picks import flat_eventos, load_config

    cfg = load_config()
    if fecha is None:
        fecha = fecha_a_analizar(cfg)
        if fecha is None:
            log.info("sin fechas completas pendientes de postmortem")
            return None

    datos = resultados_de_fecha(cfg, fecha, min_jugados=1)
    if datos is None:
        log.error("la fecha %d no tiene ningún partido jugado todavía", fecha)
        return None
    resultados, faltan = datos
    if faltan:
        log.warning("fecha %d: %d partido(s) sin resultado (suspendido o por jugar) — "
                    "se analiza lo jugado; regenerá el postmortem cuando se resuelvan",
                    fecha, faltan)

    mis_numeros = sorted(mis_numeros_env())
    eventos_fecha = [ev for ev in flat_eventos(cfg) if ev["fecha_n"] == fecha]

    # Antes de leer la planilla, re-adoptar lo que la web dice de los partidos
    # cerrados. La adopción del drift de las 23:50 puede quedar SOMBREADA por un
    # rerun en vuelo (arranca 23:35, lee la latest pre-adopción y escribe 30-60
    # min después): sin esta pasada, el postmortem atribuía puntos con picks que
    # nunca se jugaron — y no se rehace, porque su condición de regeneración mira
    # resultados nuevos, no cambios de planilla.
    if not dry_run:
        try:
            from src.clausura.drift_audit import adoptar_picks_cerrados, fetch_cargados
            cargados = fetch_cargados(cfg["pencas"]["paga"]["id"], set(mis_numeros))
            adoptar_picks_cerrados(cargados, mis_numeros, eventos_fecha,
                                   datetime.now(timezone.utc))
        except Exception as e:                                 # noqa: BLE001
            log.warning("no pude re-adoptar los picks reales antes del postmortem "
                        "(%s) — sigo con la planilla guardada", e)

    picks, e_pts = picks_de_planilla(fecha, mis_numeros)
    pool = latest_snapshot_participaciones()

    st = compute_stats(fecha, eventos_fecha, resultados, picks, e_pts,
                       pool, set(mis_numeros))
    premios = {p["tipo"]: p["monto"] for p in cfg.get("premios", [])}
    reporte = formatear_postmortem(st, premio_fecha=premios.get("FECHA"), faltan=faltan)

    # Chequeo de asignación: fecha (desde el snapshot) + temporada (desde el ranking).
    asig_fecha = chequeo_asignacion(st)
    asig_temp = None
    ranking_pts: dict[int, int] = {}
    try:
        with PencaApiClient() as api:
            ranking = api.ranking(cfg["pencas"]["paga"]["id"])
        ranking_pts = {r.numero_participacion: r.puntos_totales for r in ranking}
        asig_temp = chequeo_asignacion_temporada(ranking_pts, set(mis_numeros))
    except Exception as e:                                     # noqa: BLE001
        log.warning("ranking no disponible (%s) — sin asignación de temporada ni tripwire", e)
    seccion = formatear_asignacion(asig_fecha, asig_temp)
    if seccion:
        reporte += "\n\n" + seccion

    # Tripwire de liquidación: nuestros puntos calculados vs los que la web publica.
    try:
        if not ranking_pts:
            raise RuntimeError("sin ranking")
        publicados = {n: p for n, p in ranking_pts.items() if n in set(mis_numeros)}
        totales = _totales_calculados(fecha, st.puntos)
        if totales is None:
            log.info("tripwire de puntos omitido: falta el postmortem de una fecha previa")
        else:
            difs = comparar_puntos_publicados(totales, publicados)
            if difs:
                reporte += ("\n\n⚠️ <b>Nuestros puntos calculados ≠ ranking de la web</b>"
                            " — ¿liquidación distinta al kernel, o especiales liquidados?\n"
                            + "\n".join(difs))
    except Exception as e:                                     # noqa: BLE001
        log.warning("tripwire de puntos vs ranking falló (%s)", e)

    # PIT del pool: ¿el modelo genera una cola tan gorda como la real? Es el único
    # observable que puede convertir "los niveles no son creíbles" en un factor de
    # corrección medido — y si la cola está corta, TODOS los deltas de
    # diferenciación quedan sesgados en la misma dirección.
    try:
        from src.clausura.pool_pit import acumular, correr_fecha, formatear
        from src.clausura.pool_snapshot import load_latest_snapshot

        snap = load_latest_snapshot()
        if snap and st.pool_puntos:
            pit = correr_fecha(fecha, cfg, snap, resultados, st.pool_puntos,
                               set(mis_numeros))
            if pit is not None:
                # Acumulado: el PIT de UNA fecha no dice nada (es un percentil de una
                # muestra de 1). La señal es la tendencia sobre 3-4 fechas.
                todos = acumular(pit, persistir=not dry_run)
                reporte += "\n\n" + formatear(todos)
    except Exception as e:                                     # noqa: BLE001
        log.warning("PIT del pool falló (%s) — el postmortem sigue sin esa sección", e)

    print(reporte.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    if not dry_run:
        PM_DIR.mkdir(parents=True, exist_ok=True)
        pm_path(fecha).write_text(json.dumps({
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "fecha": fecha,
            "resultados": {str(k): list(v) for k, v in st.resultados.items()},
            "puntos": st.puntos,
            "exactos": st.exactos,
            "esperados": st.esperados,
            "pool_puntos": st.pool_puntos,
            "pool_exactos_por_evento": {str(k): v for k, v in st.pool_exactos_por_evento.items()},
            "asignacion": {"fecha": asig_fecha, "temporada": asig_temp},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        from src.notifier.telegram import TelegramConfig, TelegramNotifier
        TelegramNotifier(TelegramConfig.from_env()).send(reporte)
    return reporte


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", type=int, default=None,
                    help="forzar una fecha puntual (default: detectar la pendiente)")
    ap.add_argument("--dry-run", action="store_true",
                    help="imprime el reporte sin Telegram ni archivo de estado")
    args = ap.parse_args()
    run(fecha=args.fecha, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""Vista del pool: dónde estamos nosotros dentro de las ~700 participaciones.

Todo esto sale de UNA request: `GET front/pencas/{id}/ranking?size=1000` devuelve
las 692 filas con puntos, puntos de la fecha y exactos por participación. No hace
falta el escaneo caro (2 requests por participación) — ése es para ver los PICKS
ajenos (Capa 5), no para ver la tabla.

Dos sutilezas del ranking que cambian lo que se muestra:

1. `posicionGeneral` NO es el puesto: es el índice del ESCALÓN de puntaje (medido
   2026-08-08 — 146 participaciones con 8 pts tienen todas posicionGeneral=1, y las
   de 3 pts tienen 2, no 147). Mostrarlo como puesto diría "vas 2°" con 146 arriba.
   Acá se calcula el puesto de competencia (1 + cuántas tienen más puntos) y se
   guarda el escalón aparte.

2. El premio general se REPARTE entre empatados en el tope (Art. 7a, ver
   src/clausura/economics.py) y no hay desempate. Entonces "ir primero" no es un
   booleano: lo que importa es cuántas cabezas hay en el tope y cuántas son
   nuestras. De ahí sale `cobro_hoy`, que es lo que cobraríamos si el campeonato
   terminara con la tabla de este momento.

La trayectoria se arma sin API extra: cada fetch real del ranking apendea una línea
a data/pool_history/clausura/ranking.jsonl (throttleada), así el histórico se
alimenta solo de las lecturas que el dashboard ya hacía.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
HIST_PATH = ROOT / "data" / "pool_history" / "clausura" / "ranking.jsonl"

# Cada cuánto se apendea una línea al histórico (además de siempre que el líder o
# nuestra mejor se muevan). Sin esto, un dashboard abierto todo el día escribiría
# una línea cada 2 minutos.
HIST_MIN_GAP_H = 3.0
HIST_MAX_FILAS = 500

# Deltas para "cuántos rivales están a ≤N puntos del líder" (amenazas al premio).
DELTAS_AMENAZA = (0, 1, 2, 3, 5)


def _percentil(valor: int, valores: list[int]) -> float:
    """% del pool con estrictamente MENOS puntos que `valor`."""
    if not valores:
        return 0.0
    return 100.0 * sum(1 for v in valores if v < valor) / len(valores)


def resumen_pool(
    rows,
    mis_numeros: set[int] | frozenset[int],
    premio_penca: float = 350_000.0,
    premio_fecha: float = 10_000.0,
) -> dict:
    """Estado competitivo del pool desde las filas crudas del ranking.

    `rows` son RankingRow (src.clausura.api). Función pura: no toca red ni disco.
    """
    if not rows:
        return {"ok": False, "error": "ranking vacío", "total": 0}

    mis_numeros = set(mis_numeros)
    pts = [r.puntos_totales for r in rows]
    n = len(rows)
    ordenados = sorted(rows, key=lambda r: (-r.puntos_totales, -r.cant_resultados_exactos))

    lider_pts = max(pts)
    en_tope = [r for r in rows if r.puntos_totales == lider_pts]
    mias_en_tope = [r for r in en_tope if r.numero_participacion in mis_numeros]
    cobro_hoy = premio_penca * len(mias_en_tope) / len(en_tope) if en_tope else 0.0

    # --- nuestras participaciones ---
    mias = []
    for r in ordenados:
        if r.numero_participacion not in mis_numeros:
            continue
        mas_pts = [o for o in rows if o.puntos_totales > r.puntos_totales]
        empatadas = [o for o in rows
                     if o.puntos_totales == r.puntos_totales
                     and o.participacion_id != r.participacion_id]
        mias.append({
            "numero": r.numero_participacion,
            "participacion_id": r.participacion_id,
            "puntos": r.puntos_totales,
            "puntos_fecha": r.puntos_por_fecha,
            "exactos": r.cant_resultados_exactos,
            "puesto": len(mas_pts) + 1,
            "escalon": r.posicion_general,
            "rivales_adelante": sum(1 for o in mas_pts
                                    if o.numero_participacion not in mis_numeros),
            "empatados": len(empatadas),
            "empatados_rivales": sum(1 for o in empatadas
                                     if o.numero_participacion not in mis_numeros),
            "gap_lider": lider_pts - r.puntos_totales,
            "percentil": round(_percentil(r.puntos_totales, pts), 1),
            "en_tope": r.puntos_totales == lider_pts,
        })

    mejor = mias[0] if mias else None
    peor = mias[-1] if mias else None

    # --- distribución de puntos, con las nuestras marcadas ---
    from collections import Counter
    dist_c = Counter(pts)
    mias_c = Counter(r.puntos_totales for r in rows if r.numero_participacion in mis_numeros)
    max_c = max(dist_c.values()) if dist_c else 1
    distribucion = [
        {
            "puntos": p,
            "count": c,
            "pct": round(100.0 * c / n, 1),
            "bar_pct": round(100.0 * c / max_c),
            "mias": mias_c.get(p, 0),
        }
        for p, c in sorted(dist_c.items(), key=lambda kv: -kv[0])
    ]

    # --- amenazas: cuántos rivales están a ≤N del líder ---
    amenazas = [
        {"delta": d,
         "rivales": sum(1 for r in rows
                        if r.numero_participacion not in mis_numeros
                        and r.puntos_totales >= lider_pts - d)}
        for d in DELTAS_AMENAZA
    ]

    # --- premio por fecha (Art. 8: solo los 8 partidos de la fecha, sin acumular) ---
    pts_fecha = [r.puntos_por_fecha for r in rows]
    lider_fecha = max(pts_fecha) if pts_fecha else 0
    tope_fecha = [r for r in rows if r.puntos_por_fecha == lider_fecha]
    mias_tope_fecha = [r for r in tope_fecha if r.numero_participacion in mis_numeros]
    mis_pts_fecha = [r.puntos_por_fecha for r in rows if r.numero_participacion in mis_numeros]
    # Con la fecha sin liquidar el API manda 0 para TODOS; sin este corte el "tope"
    # sería el pool entero y saldría un cobro de $14 por cabeza que no existe.
    fecha_activa = lider_fecha > 0
    fecha = {
        "activo": fecha_activa,
        "lider": lider_fecha,
        "empatados": len(tope_fecha) if fecha_activa else 0,
        "mias_en_tope": len(mias_tope_fecha) if fecha_activa else 0,
        "nuestra_mejor": max(mis_pts_fecha) if (fecha_activa and mis_pts_fecha) else None,
        "cobro_hoy": (premio_fecha * len(mias_tope_fecha) / len(tope_fecha)
                      if fecha_activa and tope_fecha else 0.0),
        "premio": premio_fecha,
    }

    exactos = [r.cant_resultados_exactos for r in rows]
    mis_exactos = [r.cant_resultados_exactos for r in rows
                   if r.numero_participacion in mis_numeros]
    mis_pts = [r.puntos_totales for r in rows if r.numero_participacion in mis_numeros]

    return {
        "ok": True,
        "total": n,
        "mias_encontradas": len(mias),
        "lider": {
            "puntos": lider_pts,
            "empatados": len(en_tope),
            "mias_en_tope": len(mias_en_tope),
            "premio_por_cabeza": premio_penca / len(en_tope) if en_tope else 0.0,
            "cobro_hoy": cobro_hoy,
            "premio": premio_penca,
        },
        "mias": mias,
        "mejor": mejor,
        "peor": peor,
        "distribucion": distribucion,
        "amenazas": amenazas,
        "fecha": fecha,
        "puntos": {
            "lider": lider_pts,
            "mediana": sorted(pts)[n // 2],
            "media": round(sum(pts) / n, 2),
            "nuestra_media": round(sum(mis_pts) / len(mis_pts), 2) if mis_pts else None,
            "nuestra_mejor": max(mis_pts) if mis_pts else None,
            "nuestra_peor": min(mis_pts) if mis_pts else None,
        },
        "exactos": {
            # El contador lo liquida un job al cierre de la fecha, no viene en vivo
            # (ver observed_exact_rate_from_ranking): 0 exactos con puntos ya
            # cargados significa "todavía no liquidó", no "nadie pegó ninguno".
            "sin_liquidar": (max(exactos) == 0 and lider_pts > 0) if exactos else False,
            "pool_max": max(exactos) if exactos else 0,
            "pool_media": round(sum(exactos) / n, 2),
            "nuestra_media": (round(sum(mis_exactos) / len(mis_exactos), 2)
                              if mis_exactos else None),
            "nuestra_mejor": max(mis_exactos) if mis_exactos else None,
        },
    }


# -------------------- histórico (trayectoria sin API extra) --------------------

def entrada_historica(
    resumen: dict,
    ts: datetime | None = None,
    liquidables: list[int] | None = None,
) -> dict:
    """La línea que se guarda: lo mínimo para dibujar la trayectoria.

    `mias` (numero/puntos/puesto de cada participación nuestra, claves cortas) y
    `liq` (evento_ids liquidables al momento de la foto, ver liquidables_en) son
    los dos campos que permiten reconstruir cuánto subió o bajó cada penca tras
    cada partido — el diff lo hace movimientos_por_partido, acá solo se fotografía.
    """
    mejor = resumen.get("mejor") or {}
    return {
        "ts": (ts or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "total": resumen.get("total"),
        "lider": resumen["lider"]["puntos"],
        "empatados_tope": resumen["lider"]["empatados"],
        "mediana": resumen["puntos"]["mediana"],
        "mejor_puntos": mejor.get("puntos"),
        "mejor_puesto": mejor.get("puesto"),
        "mejor_numero": mejor.get("numero"),
        "mias_en_tope": resumen["lider"]["mias_en_tope"],
        "cobro_hoy": round(resumen["lider"]["cobro_hoy"]),
        "exactos_max": resumen["exactos"]["pool_max"],
        "mias": [{"n": m["numero"], "p": m["puntos"], "pu": m["puesto"]}
                 for m in resumen.get("mias", [])],
        "liq": sorted(liquidables or []),
    }


def _vale_la_pena(nueva: dict, ultima: dict | None, min_gap_h: float) -> bool:
    """Se guarda si pasó el intervalo O si cambió algo que importa.

    Sin la segunda condición un movimiento de tabla a las 2h del último guardado se
    perdería; sin la primera, una pestaña abierta escribiría una línea por refresh."""
    if ultima is None:
        return True
    for k in ("lider", "mejor_puntos", "mejor_puesto", "total"):
        if nueva.get(k) != ultima.get(k):
            return True
    # Cualquier movimiento de puesto/puntos de las nuestras también amerita foto:
    # sin esto, una liquidación que solo mueve participaciones del medio de la
    # tabla no quedaría registrada y el diff por partido saldría agujereado.
    if nueva.get("mias") and ultima.get("mias") and nueva["mias"] != ultima["mias"]:
        return True
    try:
        prev = datetime.fromisoformat(ultima["ts"])
    except Exception:
        return True
    return datetime.now(timezone.utc) - prev >= timedelta(hours=min_gap_h)


def leer_historia(path: Path | None = None, limite: int = HIST_MAX_FILAS) -> list[dict]:
    p = path or HIST_PATH
    if not p.exists():
        return []
    out = []
    for linea in p.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea:
            continue
        try:
            out.append(json.loads(linea))
        except json.JSONDecodeError:
            continue                      # línea a medio escribir: se ignora
    return out[-limite:]


def historia_por_fecha(pm_dir: Path | None = None) -> list[dict]:
    """Fecha a fecha desde los postmortems ya guardados en disco (cero API).

    El postmortem de cada fecha (src/clausura/postmortem.py) deja nuestros puntos y
    exactos por participación más los puntos del pool en esa fecha: alcanza para
    decir si ganamos el premio de la fecha y en qué percentil caímos.
    """
    d = pm_dir or (ROOT / "data" / "postmortems" / "clausura")
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("fecha_*.json")):
        try:
            pm = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        mis_pts = [int(v) for v in (pm.get("puntos") or {}).values()]
        mis_ex = [int(v) for v in (pm.get("exactos") or {}).values()]
        pool = [int(v) for v in (pm.get("pool_puntos") or [])]
        if not mis_pts:
            continue
        mejor = max(mis_pts)
        pool_max = max(pool) if pool else None
        out.append({
            "fecha": pm.get("fecha"),
            "nuestra_mejor": mejor,
            "nuestra_media": round(sum(mis_pts) / len(mis_pts), 1),
            "nuestros_exactos": sum(mis_ex),
            "pool_max": pool_max,
            "pool_mediana": sorted(pool)[len(pool) // 2] if pool else None,
            "n_rivales": len(pool),
            # ganamos la fecha si nadie del pool observado sacó más (empate = se reparte)
            "ganamos_fecha": (pool_max is not None and mejor >= pool_max),
            "percentil": (round(100.0 * sum(1 for p_ in pool if p_ < mejor) / len(pool), 1)
                          if pool else None),
        })
    return out


# Un partido no puede estar liquidado antes de terminar: inicio + 105' (90' +
# descuentos + entretiempo) es la cota inferior. Filtra el caso sábado en cascada:
# a la hora en que liquida el partido de la mañana, el del mediodía ya CERRÓ pero
# no pudo haber terminado — sin este filtro el diff se lo atribuiría a los dos.
LIQUIDABLE_TRAS_MIN = 105


def liquidables_en(eventos: list[dict], ts: datetime) -> list[int]:
    """Evento_ids cuyo puntaje YA PUDO estar liquidado en el ranking a las `ts`.

    Criterio sin API extra (el estado FINALIZADO vivo no está en el config):
    cierre pasado Y arrancó hace ≥105 min. Es una cota superior deliberada —
    la liquidación real puede demorar más; el diff entre fotos consecutivas
    atribuye el movimiento a los eventos NUEVOS de este conjunto.
    """
    out = []
    for ev in eventos:
        try:
            inicio = datetime.fromisoformat(ev["inicio_utc"])
            cierre = datetime.fromisoformat(ev["cierre_pronostico_utc"])
        except (KeyError, ValueError):
            continue
        if cierre <= ts and (ts - inicio).total_seconds() >= LIQUIDABLE_TRAS_MIN * 60:
            out.append(ev["evento_id"])
    return sorted(out)


def movimientos_por_partido(
    historia: list[dict],
    eventos: list[dict],
    limite: int = 10,
) -> list[dict]:
    """Cuánto subió o bajó cada participación nuestra tras cada partido liquidado.

    Diff entre fotos consecutivas del histórico que traen `mias`: si los puntos de
    alguna nuestra cambiaron, el movimiento se atribuye a los eventos que entraron
    a `liq` entre una foto y la otra (normalmente UNO; si dos liquidan en la misma
    ventana salen juntos, y honesto es mostrarlos juntos). Los cambios de puesto
    SIN puntos nuevos nuestros salen como fila propia etiquetada "reacomodo" —
    el puesto es relativo y lo mueven también los puntos ajenos.
    """
    nombre = {ev["evento_id"]: f"{ev.get('local', '?')} vs {ev.get('visitante', '?')}"
              for ev in eventos}
    fotos = [h for h in historia if h.get("mias")]
    out: list[dict] = []
    for antes, ahora in zip(fotos, fotos[1:]):
        prev = {m["n"]: m for m in antes["mias"]}
        movs = []
        hubo_puntos = False
        for m in ahora["mias"]:
            p = prev.get(m["n"])
            if p is None:
                continue
            if m["pu"] != p["pu"] or m["p"] != p["p"]:
                movs.append({
                    "numero": m["n"],
                    "puesto_antes": p["pu"],
                    "puesto_despues": m["pu"],
                    "delta_puesto": p["pu"] - m["pu"],     # >0 = subió
                    "puntos_ganados": m["p"] - p["p"],
                })
                hubo_puntos = hubo_puntos or m["p"] != p["p"]
        if not movs:
            continue
        nuevos = sorted(set(ahora.get("liq") or []) - set(antes.get("liq") or []))
        partidos = [nombre.get(eid, f"evento {eid}") for eid in nuevos]
        out.append({
            "ts": ahora["ts"],
            "partidos": partidos if (partidos and hubo_puntos) else [],
            "etiqueta": (" + ".join(partidos) if (partidos and hubo_puntos)
                         else "reacomodo del pool (sin puntos nuevos nuestros)"),
            "movimientos": sorted(movs, key=lambda x: -x["delta_puesto"]),
            "subieron": sum(1 for x in movs if x["delta_puesto"] > 0),
            "bajaron": sum(1 for x in movs if x["delta_puesto"] < 0),
        })
    return out[-limite:][::-1]


def registrar_historia(
    resumen: dict,
    path: Path | None = None,
    min_gap_h: float = HIST_MIN_GAP_H,
    liquidables: list[int] | None = None,
) -> bool:
    """Apendea la foto del pool si corresponde. Devuelve True si escribió.

    Nunca propaga errores: el histórico es un extra del dashboard, no puede tumbar
    la página si el disco está lleno o el FS es read-only.
    """
    if not resumen.get("ok"):
        return False
    p = path or HIST_PATH
    try:
        historia = leer_historia(p, limite=1)
        nueva = entrada_historica(resumen, liquidables=liquidables)
        if not _vale_la_pena(nueva, historia[-1] if historia else None, min_gap_h):
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(nueva, ensure_ascii=False) + "\n")
        return True
    except Exception as e:                          # noqa: BLE001 — extra, no crítico
        log.warning("no pude guardar el histórico del pool: %s", e)
        return False

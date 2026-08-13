"""Detector de bloques de compra: ¿hay OTRO sistema jugando esta penca?

EL HALLAZGO QUE LO HABILITA (auditoría 13/8): numeroParticipacion = participacionId
+ 899.190.274, con ids secuenciales GLOBALES de Supermatch. Quien compra varias
participaciones juntas queda con ids casi consecutivos — un "bloque de compra".
Nuestras 12 (una sola compra) quedaron en el rango 68574-68592 con huecos de hasta
6 ids (los del medio se fueron a otras pencas), así que la segmentación tolera
huecos. Medido en el snapshot real v9: 727 participaciones, runs estrictos de hasta
43 consecutivos.

QUÉ BUSCA. No copias (eso ya lo hace el análisis de clones: 37/686 en 8 clusters).
Busca OPTIMIZADORES: bloques cuyos picks son raros-pero-inteligentes y que se
diversifican INTERNAMENTE a propósito — la firma opuesta a la del fan que compra 5
boletas y las llena casi iguales. Cuatro features por bloque de tamaño ≥ 4:

  * rareza          media de −log10 Q_emp(pick), con Q leave-block-out (los picks
                    del propio bloque no infl an su Q — importa en los bloques grandes)
  * similitud       fracción media de partidos en que dos miembros coinciden;
                    clones: ALTA · optimizador: BAJA (anti-correlación deliberada)
  * cobertura       fracción de partidos donde la unión del bloque incluye un pick
                    que el pool casi no juega (frec < 3%) o el 0-0
  * calidad_raros   E[pts] bajo la grilla del modelo de los picks RAROS del bloque;
                    el hobbista raro pica 3-2 y 4-1, el optimizador pica el raro
                    que vale — requiere grillas (ratings), si no hay se omite

Cada feature se convierte en percentil contra una distribución NULA: B subconjuntos
al azar del mismo tamaño. El score compuesto promedia los percentiles en la
dirección "optimizador" (rareza↑, similitud↓, cobertura↑, calidad↑). Un bloque con
similitud ≥ p99 se marca CLON (otra especie, ya conocida).

CALIBRACIÓN OBLIGATORIA: nuestro propio bloque tiene que salir arriba. Si el
detector no nos encuentra a nosotros, no sirve — se imprime marcado con ►.

CUÁNDO CONCLUIR. Con una sola fecha observada (~7 partidos) un bloque "raro" puede
ser casualidad: el reporte se niega a concluir con menos de 3 fechas de picks
observados y lo dice. La corrida útil es con F2-F4 acumuladas en los snapshots.

QUÉ SE HACE CON EL RESULTADO (decidido en la auditoría): si aparece un bloque
optimizador, el paso siguiente es modelarlo como rival PERSISTENTE en
rivals.sample_picks_match (bootstrap de su patrón, no i.i.d. ∝ Q^γ) — medido con
deltas pareados. NO se cambian nuestros picks: el gate por partido ya impide la
copia y el Art. 7a hace que la copia masiva se autodestruya.

Uso (en el VPS, que tiene los snapshots reales):
    python -m scripts.detector_bloques
    python -m scripts.detector_bloques --min-tam 4 --tolerancia-gap 6 --top 15
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.clausura.economics import N_SCORES, index_score, score_index  # noqa: E402

# Huecos de id tolerados dentro de un bloque. 6 = el hueco máximo observado DENTRO
# de nuestra propia compra única (ids globales: los del medio son de otras pencas).
# Más tolerancia junta compradores ajenos; eso DILUYE las features hacia la media
# del pool — sesga a perder bloques, no a inventarlos, que es el lado sano.
TOLERANCIA_GAP = 6
MIN_TAM = 4
# La tolerancia ENCADENA: cerca del lock la tasa de compras sube y los huecos se
# achican, así que la cadena junta decenas de compradores (medido en el v9 real:
# un "bloque" de 178). Un run así es una ventana de compras masiva, no una persona:
# se reporta como tal y se excluye del scoring.
MAX_TAM = 40
B_NULA = 400
FREC_RARA = 0.03          # un pick es "raro" si el pool lo juega < 3% en ese partido
SEED = 20260813


@dataclass
class Bloque:
    ids: list[int]
    numeros: list[int]
    filas: list[dict]                 # las participaciones del snapshot
    es_nuestro: bool = False
    features: dict = field(default_factory=dict)
    percentiles: dict = field(default_factory=dict)
    score: float = 0.0
    score_raro: float = 0.0
    score_ev: float = 0.0
    es_clon: bool = False

    @property
    def tam(self) -> int:
        return len(self.ids)

    @property
    def rango(self) -> str:
        return f"{min(self.ids)}-{max(self.ids)}"


# -------------------- segmentación --------------------

def segmentar(participaciones: list[dict], tolerancia: int = TOLERANCIA_GAP,
              mis_numeros: set[int] | None = None) -> list[Bloque]:
    """Bloques de compra: ids ordenados, cortando donde el hueco supera la tolerancia.

    NUESTRO bloque se talla EXACTO por número antes de segmentar: la membresía es
    conocida (CLAUSURA_MIS_PARTICIPACIONES), no hay que adivinarla con la
    tolerancia. Sin esto, la primera corrida real fusionó nuestras 12 con ~17
    compradores vecinos (bloque de 29): los extraños diluían las features y la
    calibración salía #6 en vez de arriba — el detector se invalidaba a sí mismo.
    """
    mis = mis_numeros or set()
    nuestras = [p for p in participaciones if int(p["numero"]) in mis]
    resto = [p for p in participaciones if int(p["numero"]) not in mis]

    orden = sorted(resto, key=lambda p: int(p["participacion_id"]))
    bloques: list[list[dict]] = []
    for p in orden:
        if bloques and (int(p["participacion_id"])
                        - int(bloques[-1][-1]["participacion_id"])) <= tolerancia:
            bloques[-1].append(p)
        else:
            bloques.append([p])

    out = []
    if nuestras:
        out.append(Bloque(
            ids=[int(p["participacion_id"]) for p in nuestras],
            numeros=[int(p["numero"]) for p in nuestras],
            filas=nuestras, es_nuestro=True))
    for filas in bloques:
        out.append(Bloque(
            ids=[int(p["participacion_id"]) for p in filas],
            numeros=[int(p["numero"]) for p in filas],
            filas=filas))
    return out


# -------------------- features --------------------

def _picks_de(fila: dict) -> dict[int, int]:
    return {int(k): score_index(min(int(v[0]), 5), min(int(v[1]), 5))
            for k, v in (fila.get("picks") or {}).items()}


def conteos_totales(participaciones: list[dict]) -> dict[int, np.ndarray]:
    counts: dict[int, np.ndarray] = {}
    for fila in participaciones:
        for eid, idx in _picks_de(fila).items():
            if eid not in counts:
                counts[eid] = np.zeros(N_SCORES)
            counts[eid][idx] += 1
    return counts


def features_bloque(
    filas: list[dict],
    counts_total: dict[int, np.ndarray],
    grids_eptos: dict[int, np.ndarray] | None = None,
) -> dict:
    """Las cuatro features de un conjunto de participaciones.

    `counts_total` son los conteos del pool ENTERO; acá se les resta el bloque
    (leave-block-out) para que sus propios picks no se expliquen a sí mismos.
    `grids_eptos[eid][idx]` = E[pts] del pick idx en el partido eid (opcional).
    """
    picks = [_picks_de(f) for f in filas]

    # leave-block-out
    counts = {eid: c.copy() for eid, c in counts_total.items()}
    for pk in picks:
        for eid, idx in pk.items():
            if eid in counts:
                counts[eid][idx] -= 1

    rarezas, calidad_raros, calidad_media, eventos = [], [], [], set()
    for pk in picks:
        for eid, idx in pk.items():
            eventos.add(eid)
            if grids_eptos and eid in grids_eptos:
                calidad_media.append(float(grids_eptos[eid][idx]))
            c = counts.get(eid)
            n = float(c.sum()) if c is not None else 0.0
            if c is None or n <= 0:
                continue
            frec = max(float(c[idx]) / n, 0.5 / n)      # 0 observados ≈ media obs.
            rarezas.append(-np.log10(frec))
            if frec < FREC_RARA and grids_eptos and eid in grids_eptos:
                calidad_raros.append(float(grids_eptos[eid][idx]))

    # similitud interna: pares sobre partidos en común
    sims = []
    for i in range(len(picks)):
        for j in range(i + 1, len(picks)):
            comunes = set(picks[i]) & set(picks[j])
            if comunes:
                sims.append(np.mean([picks[i][e] == picks[j][e] for e in comunes]))

    # cobertura conjunta de lo que el pool evita
    cubiertos = 0
    idx_00 = score_index(0, 0)
    for eid in eventos:
        c = counts.get(eid)
        if c is None or c.sum() <= 0:
            continue
        frecs = c / c.sum()
        union = {pk[eid] for pk in picks if eid in pk}
        if any(idx == idx_00 or frecs[idx] < FREC_RARA for idx in union):
            cubiertos += 1

    return {
        "rareza": float(np.mean(rarezas)) if rarezas else np.nan,
        "similitud": float(np.mean(sims)) if sims else np.nan,
        "cobertura": cubiertos / len(eventos) if eventos else np.nan,
        "calidad_raros": float(np.mean(calidad_raros)) if calidad_raros else np.nan,
        "calidad_media": float(np.mean(calidad_media)) if calidad_media else np.nan,
        "n_eventos": len(eventos),
    }


# -------------------- nula y score --------------------

# Dos ESPECIES de sistema, dos perfiles. La calibración real del 13/8 refutó el
# perfil único: nuestro portfolio no es "raro y diverso" — el menú K_EV juega los
# top-5 por E[pts], pegados al modo. Nuestra huella es CALIDAD máxima en todos los
# picks, repartida entre las mejores celdas (similitud media-baja). Un cazador de
# varianza tiene la otra huella: raro-pero-bueno con cobertura deliberada.
DIRECCION = {"rareza": +1, "similitud": -1, "cobertura": +1,
             "calidad_raros": +1, "calidad_media": +1}
PERFIL_RARO = {"rareza": +1, "similitud": -1, "cobertura": +1, "calidad_raros": +1}
# Perfil EV = calidad media SOLA: castigar similitud acá confundía — la peña
# chalk que copia el modo ya la atrapa la marca de CLON (similitud absoluta),
# y nuestro propio portfolio reparte picks entre pocas celdas top (sim media).
PERFIL_EV = {"calidad_media": +1}


def score_bloques(
    bloques: list[Bloque],
    participaciones: list[dict],
    grids_eptos: dict[int, np.ndarray] | None = None,
    b_nula: int = B_NULA,
    seed: int = SEED,
    min_tam: int = MIN_TAM,
    max_tam: int = MAX_TAM,
) -> list[Bloque]:
    """Percentiles contra subconjuntos al azar del mismo tamaño + score compuesto."""
    rng = np.random.default_rng(seed)
    counts_total = conteos_totales(participaciones)

    candidatos = [b for b in bloques if min_tam <= b.tam <= max_tam]
    megas = [b for b in bloques if b.tam > max_tam]
    if megas:
        print(f"({len(megas)} run(s) de más de {max_tam} ids excluidos del scoring: "
              f"ventanas de compras masiva, no un comprador — tamaños "
              f"{sorted(b.tam for b in megas)})")
    nulas: dict[int, dict[str, np.ndarray]] = {}
    for tam in sorted({b.tam for b in candidatos}):
        muestras = []
        for _ in range(b_nula):
            filas = [participaciones[k]
                     for k in rng.choice(len(participaciones), size=tam, replace=False)]
            muestras.append(features_bloque(filas, counts_total, grids_eptos))
        nulas[tam] = {k: np.array([m[k] for m in muestras]) for k in DIRECCION}

    def _score_perfil(b: Bloque, perfil: dict) -> float:
        pcts = []
        for k, direccion in perfil.items():
            if k not in b.percentiles:
                continue
            pct = b.percentiles[k]
            pcts.append(pct if direccion > 0 else 100.0 - pct)
        return float(np.mean(pcts)) if pcts else 0.0

    for b in candidatos:
        b.features = features_bloque(b.filas, counts_total, grids_eptos)
        for k in DIRECCION:
            v = b.features[k]
            nula = nulas[b.tam][k]
            nula = nula[~np.isnan(nula)]
            if np.isnan(v) or len(nula) == 0:
                continue
            b.percentiles[k] = 100.0 * (np.mean(nula < v) + 0.5 * np.mean(nula == v))
        b.score_raro = _score_perfil(b, PERFIL_RARO)
        b.score_ev = _score_perfil(b, PERFIL_EV)
        # el score general es el peor caso PARA nosotros: cualquiera de las dos
        # especies de sistema es un rival que interesa detectar
        b.score = max(b.score_raro, b.score_ev)
        # CLON por similitud ABSOLUTA: copias reales comparten ≥80% de los picks.
        # El umbral por percentil era frágil (cualquier bloque apenas más
        # parecido que el azar saltaba como clon).
        b.es_clon = (b.features.get("similitud") or 0.0) >= 0.8
    return sorted(candidatos, key=lambda b: -b.score)


# -------------------- timestamps (si el snapshot los tiene) --------------------

def resumen_timestamps(b: Bloque) -> str | None:
    """Dispersión de carga y ediciones — la mitad del detector que arranca con los
    snapshots nuevos (PR #191). '3 min' entre 12 planillas huele a script."""
    ts, ediciones = [], 0
    for fila in b.filas:
        for eid, par in (fila.get("picks_ts") or {}).items():
            cr, lm = (par + [None, None])[:2] if isinstance(par, list) else par
            if cr:
                try:
                    ts.append(datetime.strptime(cr, "%d-%m-%Y %H:%M:%S"))
                except ValueError:
                    pass
            if cr and lm and lm != cr:
                ediciones += 1
    if not ts:
        return None
    spread = (max(ts) - min(ts)).total_seconds() / 60
    return f"carga en {spread:.0f} min · {ediciones} ediciones"


# -------------------- reporte --------------------

def fechas_observadas(participaciones: list[dict], eventos_por_fecha: int = 8) -> float:
    eids = {eid for f in participaciones for eid in (f.get("picks") or {})}
    return len(eids) / eventos_por_fecha


def formatear(bloques: list[Bloque], n_fechas: float, top: int = 12) -> str:
    lines = ["🕵️  Detector de bloques de compra — ¿hay otro sistema en la penca?", ""]
    lines.append(f"{'rango ids':>14} {'tam':>4} {'score':>6} {'perfil':>7}  "
                 f"{'rareza':>7} {'simil.':>7} {'cobert':>7} {'calRar':>7} {'calMed':>7}  etiqueta")
    for b in bloques[:top]:
        marca = "► NUESTRO" if b.es_nuestro else ("CLON" if b.es_clon else
                                                  ("⚠️ optimizador?" if b.score >= 90 else ""))
        p = b.percentiles
        perfil = "raro" if b.score_raro >= b.score_ev else "ev"

        def _c(k):
            return f"{p[k]:7.0f}" if k in p else "      —"

        lines.append(
            f"{b.rango:>14} {b.tam:>4} {b.score:>6.1f} {perfil:>7}  "
            f"{_c('rareza')} {_c('similitud')} {_c('cobertura')} "
            f"{_c('calidad_raros')} {_c('calidad_media')}  {marca}")
        ts = resumen_timestamps(b)
        if ts and (b.es_nuestro or b.score >= 85):
            lines.append(f"{'':>26}{ts}")
    lines.append("")
    lines.append("(percentiles vs subconjuntos al azar del mismo tamaño; perfil raro = "
                 "rareza↑ similitud↓ cobertura↑ calRar↑ · perfil ev = calMed↑ · score = max de los dos · CLON = similitud absoluta ≥0,8)")

    nuestro = next((b for b in bloques if b.es_nuestro), None)
    if nuestro is None:
        lines.append("\n❌ CALIBRACIÓN FALLIDA: el detector no encontró NUESTRO bloque "
                     "(¿mis_numeros? ¿tolerancia de gap?). No confiar en el resto.")
    else:
        pos = bloques.index(nuestro) + 1
        linea = (f"\ncalibración: nuestro bloque salió #{pos}/{len(bloques)} "
                 f"con score {nuestro.score:.1f} (raro {nuestro.score_raro:.0f} · "
                 f"ev {nuestro.score_ev:.0f})")
        if n_fechas < 3:
            linea += (" — con <3 fechas ni nuestra firma alcanza a formarse; "
                      "la vara de top-3 aplica recién con 3+")
        elif pos > max(3, len(bloques) // 10):
            linea += " ⚠️ esperábamos top-3: revisar features"
        else:
            linea += " ✓"
        lines.append(linea)

    if n_fechas < 3:
        lines.append(f"\n⏳ {n_fechas:.1f} fecha(s) de picks observados: INSUFICIENTE "
                     f"para concluir sobre terceros — un bloque raro puede ser "
                     f"casualidad. Re-correr con 3+ fechas (fin de agosto).")
    else:
        sospechosos = [b for b in bloques
                       if b.score >= 90 and not b.es_nuestro and not b.es_clon]
        if sospechosos:
            lines.append(f"\n⚠️ {len(sospechosos)} bloque(s) con firma de optimizador. "
                         f"Paso siguiente (pre-decidido): modelarlos como rival "
                         f"persistente en rivals.sample_picks_match, medir pareado. "
                         f"NO tocar nuestros picks.")
        else:
            lines.append("\n✓ Ningún bloque ajeno con firma de optimizador. "
                         "Tranquilidad medida, no supuesta.")
    return "\n".join(lines)


# -------------------- grillas de E[pts] (opcionales) --------------------

def grids_eptos_de_ratings() -> dict[int, np.ndarray] | None:
    """E[pts] por (evento, pick) bajo las grillas PREDICTIVAS de ratings.

    Solo ratings, sin odds: los partidos que el detector mira ya se jugaron y sus
    cuotas no existen más; las grillas de ratings son la vara estable de "qué pick
    raro tenía valor". Si algo falla, la feature se omite y el detector sigue.
    """
    try:
        from src.clausura.economics import flatten_grid, points_matrix
        from src.clausura.picks import build_season_grids, ensure_ratings, flat_eventos, load_config

        cfg = load_config()
        eventos = flat_eventos(cfg)
        _, _, pred_grids, _ = build_season_grids(eventos, ensure_ratings(), {}, {})
        pm = points_matrix(False).astype(float)
        out = {}
        for ev, g in zip(eventos, pred_grids):
            q = flatten_grid(g)
            out[ev["evento_id"]] = pm @ q          # E[pts] de cada pick idx
        return out
    except Exception as e:                                     # noqa: BLE001
        print(f"(sin grillas de E[pts]: {e} — la feature calidad_raros se omite)")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-tam", type=int, default=MIN_TAM)
    ap.add_argument("--tolerancia-gap", type=int, default=TOLERANCIA_GAP)
    ap.add_argument("--max-tam", type=int, default=MAX_TAM)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--b-nula", type=int, default=B_NULA)
    args = ap.parse_args()

    from src.clausura.pool_snapshot import load_latest_snapshot
    from src.clausura.rivals import mis_numeros_env

    snap = load_latest_snapshot()
    if not snap:
        print("sin snapshot del pool — el detector necesita los picks públicos")
        raise SystemExit(1)
    participaciones = [p for p in snap.get("participaciones", []) if p.get("picks")]
    mis = set(mis_numeros_env())
    print(f"snapshot: {len(participaciones)} participaciones con picks · "
          f"{fechas_observadas(participaciones):.1f} fechas observadas\n")

    bloques = segmentar(participaciones, tolerancia=args.tolerancia_gap, mis_numeros=mis)
    print(f"bloques de tamaño ≥ {args.min_tam}: "
          f"{sum(1 for b in bloques if b.tam >= args.min_tam)} "
          f"(de {len(bloques)} bloques totales)\n")

    scored = score_bloques(bloques, participaciones,
                           grids_eptos=grids_eptos_de_ratings(), b_nula=args.b_nula,
                           min_tam=args.min_tam, max_tam=args.max_tam)
    print(formatear(scored, fechas_observadas(participaciones), top=args.top))


if __name__ == "__main__":
    main()

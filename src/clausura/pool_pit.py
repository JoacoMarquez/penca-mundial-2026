"""PIT del pool: ¿el modelo de rivales genera una cola tan gorda como la real?

El sistema se auto-reporta P(cobrar el premio grande) de 29-59% con ~725 rivales.
Que el NIVEL no sea creíble ya estaba documentado; lo que la auditoría del 13/8
señaló es por qué eso importa para las DECISIONES:

    si la vara simulada para ganar es más baja que la real, el optimizador cree que
    portfolios cuasi-chalk llegan a la cima seguido, y entonces TODOS los deltas
    "más diferenciación vs menos" quedan sesgados en contra de diferenciar.

Los deltas pareados (common random numbers) cancelan el ruido común, no el SESGO
común. Este módulo lo mide con datos que ya tenemos.

## El test

Para una fecha ya jugada, se comparan dos distribuciones de puntos-de-fecha del pool:

  * la **real**: los puntos que sacaron las ~725 participaciones (el postmortem ya
    los guarda en `pool_puntos`);
  * la **simulada**: se generan rivales como los genera el modelo —pick i.i.d. por
    partido ∝ Q^γ_r, más Bernoulli(p_show)— y se los liquida contra los resultados
    REALES de esa fecha, con el kernel de Supermatch (estrella ×2 incluida).

Fijar los resultados en los reales es deliberado: aísla el modelo de RIVALES del
modelo de resultados. Lo que se está preguntando es "dados estos resultados,
¿el pool simulado rinde como el pool real?".

## Por qué no es circular

La Q de cada partido se toma EMPÍRICA de la propia fecha, así que las marginales
por partido coinciden por construcción: la media simulada tiene que dar ≈ la media
real, y de hecho eso sirve de test de sanidad. Lo que la simulación NO puede
reproducir es la **correlación entre partidos dentro de una misma participación**:
el modelo samplea cada partido independientemente, y el rival real que le pega a
un partido tiende a pegarle a los otros (juega mirando las cuotas — puntería, que
γ no captura: γ mide cuán chalk es, no cuán acertado).

Entonces todo hueco en los cuantiles ALTOS es exactamente la cola que el modelo se
está perdiendo. Si el máximo real cae sistemáticamente arriba del máximo simulado,
la vara está corta y los rechazos de diferenciación (cobertura, hueco) hay que
re-medirlos.

`p_max` es el PIT propiamente dicho: la fracción de simulaciones cuyo máximo quedó
por debajo del máximo real. Con el modelo bien calibrado se reparte uniforme entre
fechas; pegado a 1.00 fecha tras fecha significa cola corta.

Uso:
    python -m src.clausura.pool_pit                # todas las fechas con postmortem
    python -m src.clausura.pool_pit --fecha 1
    python -m src.clausura.pool_pit --sims 4000
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.clausura.economics import N_SCORES, points_matrix, score_index

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PIT_DIR = ROOT / "data" / "pit" / "clausura"

# Sorteos del PIT. No es el simulador de producción: acá cada sorteo es una fecha
# entera de ~725 rivales, así que 2.000 ya dan cuantiles estables y corre en segundos.
DEFAULT_SIMS = 2_000

_PM = {False: points_matrix(False), True: points_matrix(True)}


@dataclass
class PitFecha:
    fecha: int
    n_rivales: int
    n_partidos: int
    reales: dict[str, float] = field(default_factory=dict)
    simulados: dict[str, float] = field(default_factory=dict)
    pit: dict[str, float] = field(default_factory=dict)

    @property
    def cola_corta(self) -> bool:
        """El máximo real quedó en el 5% superior de lo que el modelo esperaba."""
        return self.pit.get("max", 0.0) >= 0.95


# -------------------- simulación (pura) --------------------

def simular_puntos_pool(
    q_por_partido: list[np.ndarray],
    gammas: np.ndarray,
    p_show: np.ndarray,
    resultados_idx: list[int],
    preferencial: list[bool],
    n_sims: int = DEFAULT_SIMS,
    seed: int = 20260813,
) -> np.ndarray:
    """(n_sims, n_rivales) puntos de la fecha, generando rivales como el modelo.

    Cada rival r saca su pick de cada partido m de Q_m^γ_r renormalizada, carga con
    probabilidad p_show[r], y se liquida contra el resultado REAL con el kernel de
    Supermatch. Es el mismo proceso generativo de `RivalModel.sample_picks_match`
    para partidos futuros, que es como el simulador ve todo lo que falta jugar.
    """
    rng = np.random.default_rng(seed)
    R = len(gammas)
    total = np.zeros((n_sims, R), dtype=np.int32)

    for m, (q, real_idx, pref) in enumerate(zip(q_por_partido, resultados_idx, preferencial)):
        # Q^γ por rival: (R, N_SCORES) normalizada. Los γ vienen de una grilla
        # chica, así que agrupar por valor único evita R exponenciaciones.
        pts_de_pick = _PM[bool(pref)][:, real_idx].astype(np.int32)   # (N_SCORES,)
        for g in np.unique(gammas):
            filas = np.flatnonzero(gammas == g)
            w = np.power(np.maximum(q, 1e-300), float(g))
            w = w / w.sum()
            # inverse-CDF vectorizado: (n_sims, len(filas))
            u = rng.random((n_sims, len(filas)))
            picks = np.searchsorted(np.cumsum(w), u, side="right").clip(0, N_SCORES - 1)
            aporte = pts_de_pick[picks]
            show = rng.random((n_sims, len(filas))) < p_show[filas][None, :]
            total[:, filas] += np.where(show, aporte, 0)
    return total


def _stats(puntos: np.ndarray) -> dict[str, float]:
    """Estadísticos de una muestra de puntos de un pool en UNA fecha."""
    return {
        "media": float(np.mean(puntos)),
        "p50": float(np.percentile(puntos, 50)),
        "p90": float(np.percentile(puntos, 90)),
        "p99": float(np.percentile(puntos, 99)),
        "max": float(np.max(puntos)),
    }


def comparar(reales: list[int], simulados: np.ndarray) -> tuple[dict, dict, dict]:
    """(stats reales, stats simulados promedio, PIT por estadístico).

    El PIT de un estadístico T es P(T_simulado < T_real): la fracción de sorteos en
    los que el modelo produjo un pool PEOR que el que se vio. Calibrado ⇒ uniforme
    en [0,1] a lo largo de las fechas; ≈1 sistemático ⇒ el modelo se queda corto.
    """
    arr = np.asarray(reales)
    obs = _stats(arr)
    por_sim = {k: np.array([_stats(fila)[k] for fila in simulados]) for k in obs}
    sim = {k: float(np.mean(v)) for k, v in por_sim.items()}
    pit = {k: float(np.mean(por_sim[k] < obs[k])
                    + 0.5 * np.mean(por_sim[k] == obs[k]))       # corrección por empates
           for k in obs}
    return obs, sim, pit


# -------------------- armado desde los datos guardados --------------------

def q_empirica_de_fecha(
    snapshot: dict, evento_ids: list[int], mis_numeros: set[int],
) -> list[np.ndarray] | None:
    """Q empírica por partido de la fecha, desde los picks públicos del snapshot.

    Usar la marginal observada es a propósito: hace que la media simulada coincida
    con la real por construcción, y deja como único residuo la correlación entre
    partidos — que es lo que se quiere medir.
    """
    from src.clausura.pool_snapshot import empirical_counts

    counts = empirical_counts(snapshot, mis_numeros)
    qs = []
    for eid in evento_ids:
        c = counts.get(eid)
        if c is None or c.sum() == 0:
            return None
        qs.append(c / c.sum())
    return qs


def gammas_y_show(
    snapshot: dict, evento_ids: list[int], qs: list[np.ndarray], mis_numeros: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """γ por rival (fiteado FUERA de la fecha testeada) y p_show empírico.

    El γ se ajusta con los picks de los OTROS partidos que el snapshot vio: fitearlo
    sobre los mismos picks que después se testean sería darle al modelo la respuesta.
    Rival sin observaciones afuera → γ=1 (el neutro de la grilla).
    """
    from src.clausura.rivals import fit_gamma
    from src.clausura.pool_snapshot import empirical_counts

    de_fecha = set(evento_ids)
    counts_todos = empirical_counts(snapshot, mis_numeros)
    logq_otros = {eid: np.log(np.maximum(c / c.sum(), 1e-300))
                  for eid, c in counts_todos.items()
                  if eid not in de_fecha and c.sum() > 0}

    gammas, shows = [], []
    for r in snapshot.get("participaciones", []):
        if int(r.get("numero", -1)) in mis_numeros:
            continue
        picks = {int(k): v for k, v in (r.get("picks") or {}).items()}
        obs_idx, logqs = [], []
        for eid, lq in logq_otros.items():
            p = picks.get(eid)
            if p is not None:
                obs_idx.append(score_index(min(int(p[0]), 5), min(int(p[1]), 5)))
                logqs.append(lq)
        gammas.append(fit_gamma(np.array(obs_idx, dtype=np.int64),
                                np.array(logqs)) if obs_idx else 1.0)
        # p_show de la fecha: fracción de sus partidos que efectivamente cargó
        shows.append(sum(1 for eid in evento_ids if eid in picks) / max(len(evento_ids), 1))
    return np.array(gammas), np.array(shows)


def correr_fecha(
    fecha: int,
    cfg: dict,
    snapshot: dict,
    resultados: dict[int, tuple[int, int]],
    pool_puntos_reales: list[int],
    mis_numeros: set[int],
    n_sims: int = DEFAULT_SIMS,
) -> PitFecha | None:
    """PIT de una fecha ya jugada. None si faltan insumos."""
    from src.clausura.picks import flat_eventos

    eventos = [ev for ev in flat_eventos(cfg)
               if ev["fecha_n"] == fecha and ev["evento_id"] in resultados]
    if not eventos or not pool_puntos_reales:
        return None
    evento_ids = [ev["evento_id"] for ev in eventos]

    qs = q_empirica_de_fecha(snapshot, evento_ids, mis_numeros)
    if qs is None:
        log.warning("fecha %d: el snapshot no cubre todos los partidos — PIT omitido", fecha)
        return None

    gammas, p_show = gammas_y_show(snapshot, evento_ids, qs, mis_numeros)
    if len(gammas) == 0:
        return None

    # MISMA población que el lado real. `pool_puntos_reales` (postmortem) solo
    # incluye rivales con ≥1 pick de la fecha; simular también a los ~50-80 que no
    # cargaron nada (p_show = 0, total 0 seguro) arrastraba la media simulada
    # ~1-1.5 pts abajo de la real y el PIT de la media clavaba 1.00 tres fechas
    # seguidas — leído el 31/8 como "el modelo subestima al rival promedio" cuando
    # era un artefacto de comparar poblaciones distintas: condicionando a
    # p_show > 0, la media simulada calza con la real en las 4 fechas (±0.3, signo
    # mixto). Los cuantiles altos nunca se enteraron (los ceros van al fondo),
    # pero la media es el termómetro de calibración de marginales y tiene que
    # medir de verdad. Ojo: esto es SOLO del PIT — en producción los no-show valen
    # como están (existen, ocupan lugar en el pool y de verdad suman 0).
    con_pick = p_show > 0
    gammas, p_show = gammas[con_pick], p_show[con_pick]

    res_idx = [score_index(min(resultados[eid][0], 5), min(resultados[eid][1], 5))
               for eid in evento_ids]
    pref = [bool(ev["preferencial"]) for ev in eventos]

    sim = simular_puntos_pool(qs, gammas, p_show, res_idx, pref, n_sims=n_sims)
    obs, sim_stats, pit = comparar(pool_puntos_reales, sim)
    return PitFecha(fecha=fecha, n_rivales=len(gammas), n_partidos=len(eventos),
                    reales=obs, simulados=sim_stats, pit=pit)


# -------------------- reporte --------------------

def formatear(pits: list[PitFecha]) -> str:
    if not pits:
        return "PIT del pool: sin fechas con insumos completos todavía."
    lines = ["<b>🎯 PIT del pool</b> — ¿el modelo genera una cola tan gorda como la real?", ""]
    for p in pits:
        lines.append(
            f"<b>Fecha {p.fecha}</b> ({p.n_rivales} rivales, {p.n_partidos} partidos)")
        for k in ("media", "p50", "p90", "p99", "max"):
            flag = " ⚠️" if k in ("p99", "max") and p.pit[k] >= 0.95 else ""
            lines.append(f"  {k:>5}: real {p.reales[k]:5.1f} · modelo "
                         f"{p.simulados[k]:5.1f} · PIT {p.pit[k]:.2f}{flag}")
    maxs = [p.pit["max"] for p in pits]
    lines.append("")
    if len(maxs) >= 3 and float(np.mean(maxs)) >= 0.90:
        lines.append(
            f"⚠️ <b>Cola corta</b>: el máximo real quedó arriba del simulado en "
            f"{sum(m >= 0.95 for m in maxs)}/{len(maxs)} fechas (PIT medio "
            f"{np.mean(maxs):.2f}). La vara para ganar es más alta que la que cree "
            f"el optimizador ⇒ los rechazos de diferenciación (cobertura, hueco) "
            f"hay que re-medirlos.")
    elif len(maxs) < 3:
        lines.append(f"<i>{len(maxs)} fecha(s): hacen falta 3-4 para leer la "
                     f"tendencia del PIT del máximo.</i>")
    else:
        lines.append(f"Cola del máximo sin sesgo claro (PIT medio {np.mean(maxs):.2f}).")
    return "\n".join(lines)


PIT_PATH = PIT_DIR / "pit_pool.json"


def _as_dict(p: PitFecha) -> dict:
    return {"fecha": p.fecha, "n_rivales": p.n_rivales, "n_partidos": p.n_partidos,
            "reales": p.reales, "simulados": p.simulados, "pit": p.pit}


def cargar() -> list[PitFecha]:
    """PITs ya calculados de fechas anteriores."""
    if not PIT_PATH.exists():
        return []
    try:
        raw = json.loads(PIT_PATH.read_text(encoding="utf-8")).get("fechas", [])
    except Exception:                                          # noqa: BLE001
        return []
    return [PitFecha(**d) for d in raw]


def guardar(pits: list[PitFecha]) -> Path:
    PIT_DIR.mkdir(parents=True, exist_ok=True)
    PIT_PATH.write_text(
        json.dumps({"fechas": [_as_dict(p) for p in pits]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return PIT_PATH


def acumular(pit: PitFecha, persistir: bool = True) -> list[PitFecha]:
    """Suma (o reemplaza) el PIT de una fecha al historial y devuelve la serie completa.

    El PIT de UNA fecha es un percentil de una muestra de tamaño 1: no dice nada. La
    señal es la TENDENCIA — un máximo real que queda arriba del simulado fecha tras
    fecha. Por eso el reporte se arma siempre sobre el acumulado.
    """
    serie = {p.fecha: p for p in cargar()}
    serie[pit.fecha] = pit
    out = [serie[k] for k in sorted(serie)]
    if persistir:
        guardar(out)
    return out


# -------------------- CLI --------------------

def run(fecha: int | None = None, n_sims: int = DEFAULT_SIMS) -> list[PitFecha]:
    from src.clausura.picks import load_config
    from src.clausura.pool_snapshot import load_latest_snapshot
    from src.clausura.postmortem import pm_path
    from src.clausura.rivals import mis_numeros_env

    cfg = load_config()
    snapshot = load_latest_snapshot()
    if not snapshot:
        log.error("sin snapshot del pool — el PIT necesita los picks públicos")
        return []
    mis = set(mis_numeros_env())

    fechas = [fecha] if fecha else sorted(
        int(n.split()[-1]) for n in cfg["fechas"]
        if pm_path(int(n.split()[-1])).exists())

    out = []
    for f in fechas:
        p = pm_path(f)
        if not p.exists():
            log.warning("fecha %d: sin postmortem — nada que comparar", f)
            continue
        pm = json.loads(p.read_text(encoding="utf-8"))
        resultados = {int(k): (int(v[0]), int(v[1]))
                      for k, v in (pm.get("resultados") or {}).items()}
        pit = correr_fecha(f, cfg, snapshot, resultados,
                           pm.get("pool_puntos") or [], mis, n_sims=n_sims)
        if pit is not None:
            out.append(pit)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", type=int, default=None, help="default: todas las que tengan postmortem")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    args = ap.parse_args()

    pits = run(fecha=args.fecha, n_sims=args.sims)
    if args.fecha and pits:
        # una fecha puntual se SUMA al historial; el barrido completo lo reemplaza
        pits = acumular(pits[0])
    print(formatear(pits).replace("<b>", "").replace("</b>", "")
          .replace("<i>", "").replace("</i>", ""))
    if pits and not args.fecha:
        print(f"\nguardado: {guardar(pits)}")


if __name__ == "__main__":
    main()

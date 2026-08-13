"""Tests del PIT del pool (src.clausura.pool_pit).

Lo que hay que probar no es que corra, sino que DETECTE: si el pool real tiene una
cola más gorda que la que el modelo genera (rivales con puntería, correlacionados
entre partidos), el PIT del máximo tiene que irse a 1. Y si el pool real sale del
mismo proceso que el modelo, el PIT NO puede gritar — si no, sería un detector que
avisa siempre y no dice nada.
"""

from __future__ import annotations

import numpy as np

from src.clausura.economics import N_SCORES, score_index
from src.clausura.pool_pit import (
    PitFecha,
    comparar,
    formatear,
    simular_puntos_pool,
)


def _q_plano(n_top: int = 6) -> np.ndarray:
    """Q concentrada en los n_top marcadores más comunes (el pool real es así)."""
    q = np.zeros(N_SCORES)
    for gl, gv in [(1, 0), (2, 1), (1, 1), (0, 0), (2, 0), (0, 1)][:n_top]:
        q[score_index(gl, gv)] = 1.0
    return q / q.sum()


# Una fecha real son 8 partidos, y el tamaño importa para la POTENCIA del
# diagnóstico: con 3 partidos y Q concentrada en 6 marcadores, el pleno sale por
# azar ~1 vez cada 216 rivales, así que el máximo simulado ya toca el techo y
# ningún pool real puede superarlo. Con 8 partidos el pleno es (1/6)^8 y el máximo
# vive lejos del techo, que es el régimen del Clausura.
REAL_IDX = [score_index(1, 0), score_index(2, 1), score_index(0, 0),
            score_index(1, 1), score_index(2, 0), score_index(0, 1),
            score_index(1, 0), score_index(2, 1)]
PREF = [False, True, False, False, False, False, False, False]


def _setup(n_rivales: int = 200):
    qs = [_q_plano() for _ in REAL_IDX]
    gammas = np.ones(n_rivales)
    p_show = np.ones(n_rivales)
    return qs, gammas, p_show


# -------------------- mecánica --------------------

def test_simulacion_tiene_la_forma_y_es_reproducible():
    qs, g, s = _setup(50)
    a = simular_puntos_pool(qs, g, s, REAL_IDX, PREF, n_sims=30, seed=7)
    b = simular_puntos_pool(qs, g, s, REAL_IDX, PREF, n_sims=30, seed=7)
    assert a.shape == (30, 50)
    assert np.array_equal(a, b)                      # misma semilla, mismo sorteo
    assert (a >= 0).all()


def test_sin_carga_no_hay_puntos():
    """p_show=0 es el rival que no cargó: 0 puntos, no puntos del pick fantasma."""
    qs, g, _ = _setup(10)
    out = simular_puntos_pool(qs, g, np.zeros(10), REAL_IDX, PREF, n_sims=5, seed=1)
    assert out.sum() == 0


def test_la_estrella_paga_doble():
    """El partido preferencial entra al kernel con ×2 (Art. 6)."""
    qs, g, s = _setup(400)
    solo_normal = simular_puntos_pool(qs, g, s, REAL_IDX, [False] * len(REAL_IDX),
                                      n_sims=200, seed=3)
    con_estrella = simular_puntos_pool(qs, g, s, REAL_IDX, PREF, n_sims=200, seed=3)
    assert con_estrella.mean() > solo_normal.mean()


def test_gamma_alto_concentra_en_el_pick_modal():
    """γ grande = rival hiper-chalk: casi siempre el marcador más probable de Q."""
    q = np.zeros(N_SCORES)
    q[score_index(1, 0)] = 0.5
    q[score_index(2, 1)] = 0.3
    q[score_index(0, 0)] = 0.2
    chalk = simular_puntos_pool([q], np.full(100, 4.0), np.ones(100),
                                [score_index(1, 0)], [False], n_sims=100, seed=11)
    disperso = simular_puntos_pool([q], np.full(100, 0.35), np.ones(100),
                                   [score_index(1, 0)], [False], n_sims=100, seed=11)
    # con el 1-0 como resultado real, el chalk le pega mucho más seguido
    assert chalk.mean() > disperso.mean()


# -------------------- calibración: el detector no puede gritar siempre --------------------

def test_pool_generado_por_el_MISMO_proceso_no_dispara():
    """Si el pool real sale del proceso que el modelo asume, el PIT tiene que quedar
    en el medio. Un detector que avisa igual no distingue nada."""
    qs, g, s = _setup(300)
    sim = simular_puntos_pool(qs, g, s, REAL_IDX, PREF, n_sims=400, seed=21)

    pits_max = []
    for k in range(15):                      # 15 pools "reales" del mismo proceso
        real = simular_puntos_pool(qs, g, s, REAL_IDX, PREF, n_sims=1, seed=500 + k)[0]
        _, _, pit = comparar(list(real), sim)
        pits_max.append(pit["max"])
    assert 0.2 < float(np.mean(pits_max)) < 0.8


def test_medias_coinciden_por_construccion():
    """La Q es la marginal empírica: la media simulada tiene que dar ≈ la real.

    Es el test de sanidad que hace no-circular al del máximo — si las medias no
    coincidieran, un hueco en la cola podría ser solo un nivel mal calibrado.

    La tolerancia sale del error estándar de la media de un pool (no de un número
    inventado): con ~300 rivales, la media de una fecha tiene ruido propio.
    """
    qs, g, s = _setup(300)
    sim = simular_puntos_pool(qs, g, s, REAL_IDX, PREF, n_sims=300, seed=31)
    real = simular_puntos_pool(qs, g, s, REAL_IDX, PREF, n_sims=1, seed=99)[0]
    obs, sim_stats, _ = comparar(list(real), sim)
    se = float(np.std(sim, axis=1).mean() / np.sqrt(sim.shape[1]))
    assert abs(obs["media"] - sim_stats["media"]) < 3 * se


# -------------------- detección: la cola gorda tiene que aparecer --------------------

def _pool_con_punteria(n_rivales: int, frac_sharp: float, tilt: float, seed: int):
    """(puntos reales, Q empírica de ESE pool) — el mundo que el modelo no sabe simular.

    Cada rival tiene una puntería s_r: con probabilidad ∝ tilt elige el resultado
    verdadero, si no saca de Q. Los `sharp` la tienen alta en TODOS los partidos, y
    ahí está la correlación entre partidos que el sampleo i.i.d. destruye.

    Devuelve además la Q marginal empírica del pool generado, que es lo que
    producción usa: el modelo recibe la marginal CORRECTA (incluido el tilt
    promedio), así que lo único que le queda faltando es la heterogeneidad. Sin
    esto el test probaría que se detecta una Q mal estimada, que es otra cosa.
    """
    from src.clausura.economics import points_matrix

    rng = np.random.default_rng(seed)
    q = _q_plano()
    sharp = rng.random(n_rivales) < frac_sharp
    s_r = np.where(sharp, tilt, 0.0)

    counts = [np.zeros(N_SCORES) for _ in REAL_IDX]
    puntos = np.zeros(n_rivales, dtype=int)
    for m, (real_idx, pref) in enumerate(zip(REAL_IDX, PREF)):
        pm = points_matrix(bool(pref))
        for r in range(n_rivales):
            if rng.random() < s_r[r]:
                pick = real_idx                       # puntería: acierta el real
            else:
                pick = int(rng.choice(N_SCORES, p=q))
            counts[m][pick] += 1
            puntos[r] += int(pm[pick, real_idx])
    return list(puntos), [c / c.sum() for c in counts]


def test_rivales_con_punteria_disparan_el_PIT_del_maximo():
    """El mecanismo que la auditoría marcó: γ mide cuán CHALK es un rival, no cuán
    ACERTADO. Un pool con punteria heterogénea tiene un máximo que el modelo no
    alcanza, aun recibiendo la marginal por partido exacta.
    """
    n = 300
    real, q_emp = _pool_con_punteria(n, frac_sharp=0.05, tilt=0.55, seed=5)
    sim = simular_puntos_pool(q_emp, np.ones(n), np.ones(n), REAL_IDX, PREF,
                              n_sims=400, seed=41)

    obs, sim_stats, pit = comparar(real, sim)
    # la marginal la tiene bien: las medias no se despegan…
    assert abs(obs["media"] - sim_stats["media"]) < 2.0
    # …pero la cola real vive arriba de la que el modelo puede generar
    assert pit["max"] >= 0.95
    assert pit["p99"] >= 0.95


def test_sin_punteria_heterogenea_el_PIT_del_maximo_no_grita():
    """Control del test de arriba: mismo generador con tilt homogéneo (todos igual
    de acertados) NO dispara — lo que el modelo se pierde es la heterogeneidad, no
    el nivel de acierto, que la marginal ya absorbe."""
    n = 300
    real, q_emp = _pool_con_punteria(n, frac_sharp=1.0, tilt=0.15, seed=6)
    sim = simular_puntos_pool(q_emp, np.ones(n), np.ones(n), REAL_IDX, PREF,
                              n_sims=400, seed=42)
    _, _, pit = comparar(real, sim)
    assert pit["max"] < 0.95


# -------------------- reporte --------------------

def _pit(fecha: int, pit_max: float) -> PitFecha:
    stats = {"media": 8.0, "p50": 8.0, "p90": 14.0, "p99": 20.0, "max": 25.0}
    return PitFecha(fecha=fecha, n_rivales=700, n_partidos=8,
                    reales=stats, simulados=stats,
                    pit={"media": 0.5, "p50": 0.5, "p90": 0.6,
                         "p99": pit_max, "max": pit_max})


def test_reporte_grita_cola_corta_con_evidencia_repetida():
    txt = formatear([_pit(1, 0.99), _pit(2, 0.97), _pit(3, 0.98), _pit(4, 0.96)])
    assert "Cola corta" in txt and "4/4" in txt
    assert "re-medirlos" in txt


def test_reporte_no_concluye_con_una_sola_fecha():
    txt = formatear([_pit(1, 0.99)])
    assert "Cola corta" not in txt
    assert "hacen falta 3-4" in txt


def test_reporte_sin_sesgo_lo_dice():
    txt = formatear([_pit(1, 0.5), _pit(2, 0.4), _pit(3, 0.6)])
    assert "sin sesgo claro" in txt


def test_cola_corta_es_el_percentil_95():
    assert _pit(1, 0.96).cola_corta
    assert not _pit(1, 0.80).cola_corta


def test_formatear_sin_datos_no_rompe():
    assert "sin fechas" in formatear([])


# -------------------- acumulado entre fechas --------------------

def test_acumula_y_reemplaza_por_fecha(tmp_path, monkeypatch):
    """El PIT de UNA fecha es un percentil de una muestra de 1: la señal es la serie.

    Y re-correr una fecha (postmortem regenerado por un resultado corregido) tiene
    que REEMPLAZAR su entrada, no duplicarla.
    """
    from src.clausura import pool_pit

    monkeypatch.setattr(pool_pit, "PIT_DIR", tmp_path)
    monkeypatch.setattr(pool_pit, "PIT_PATH", tmp_path / "pit_pool.json")

    pool_pit.acumular(_pit(1, 0.97))
    pool_pit.acumular(_pit(3, 0.55))
    serie = pool_pit.acumular(_pit(2, 0.80))
    assert [p.fecha for p in serie] == [1, 2, 3]              # ordenado por fecha

    serie = pool_pit.acumular(_pit(1, 0.10))                  # fecha 1 re-corrida
    assert [p.fecha for p in serie] == [1, 2, 3]              # no duplica
    assert serie[0].pit["max"] == 0.10                        # y pisa el valor viejo


def test_acumular_sin_persistir_no_escribe(tmp_path, monkeypatch):
    """El --dry-run del postmortem no puede dejar rastro en el historial."""
    from src.clausura import pool_pit

    monkeypatch.setattr(pool_pit, "PIT_DIR", tmp_path)
    monkeypatch.setattr(pool_pit, "PIT_PATH", tmp_path / "pit_pool.json")
    pool_pit.acumular(_pit(1, 0.9), persistir=False)
    assert not (tmp_path / "pit_pool.json").exists()


def test_cargar_tolera_archivo_roto(tmp_path, monkeypatch):
    from src.clausura import pool_pit

    monkeypatch.setattr(pool_pit, "PIT_PATH", tmp_path / "pit_pool.json")
    (tmp_path / "pit_pool.json").write_text("{ no es json", encoding="utf-8")
    assert pool_pit.cargar() == []


# -------------------- integración: armado desde snapshot + config --------------------

def _cfg_fecha(evento_ids, pref_id):
    return {"fechas": {"Fecha 1": {"fecha_id": 280, "eventos": [
        {"evento_id": eid, "local": f"L{eid}", "visitante": f"V{eid}",
         "preferencial": eid == pref_id,
         "inicio_utc": "2026-08-08T18:00:00+00:00",
         "cierre_pronostico_utc": "2026-08-08T17:45:00+00:00"}
        for eid in evento_ids]}}}


def _snapshot(evento_ids, n_rivales=40, seed=3):
    """Pool sintético: picks de los marcadores comunes + un par de partidos EXTRA
    (fecha distinta) para que el γ se pueda fitear fuera de la fecha testeada."""
    rng = np.random.default_rng(seed)
    comunes = [(1, 0), (2, 1), (1, 1), (0, 0), (2, 0), (0, 1)]
    parts = []
    for i in range(n_rivales):
        picks = {str(eid): list(comunes[rng.integers(len(comunes))])
                 for eid in evento_ids}
        picks.update({"9001": [1, 0], "9002": [2, 1]})     # partidos de otra fecha
        parts.append({"participacion_id": i, "numero": 500 + i, "puntos": 0,
                      "exactos": 0, "picks": picks})
    return {"generado_utc": "2026-08-09T02:00:00+00:00", "participaciones": parts}


def test_correr_fecha_arma_todo_desde_los_datos_guardados():
    from src.clausura.pool_pit import correr_fecha

    eids = [101, 102, 103]
    cfg = _cfg_fecha(eids, pref_id=102)
    snap = _snapshot(eids)
    resultados = {101: (1, 0), 102: (2, 1), 103: (0, 0)}
    reales = [10, 12, 8, 20, 6] * 8

    pit = correr_fecha(1, cfg, snap, resultados, reales, mis_numeros=set(), n_sims=120)
    assert pit is not None
    assert pit.fecha == 1 and pit.n_partidos == 3 and pit.n_rivales == 40
    assert set(pit.pit) == {"media", "p50", "p90", "p99", "max"}
    assert all(0.0 <= v <= 1.0 for v in pit.pit.values())


def test_correr_fecha_sin_cobertura_del_snapshot_devuelve_none():
    """Si el snapshot no vio algún partido de la fecha no hay Q empírica: mejor no
    reportar que reportar un PIT calculado sobre una Q inventada."""
    from src.clausura.pool_pit import correr_fecha

    eids = [101, 102, 103]
    snap = _snapshot([101, 102])              # le falta el 103
    assert correr_fecha(1, _cfg_fecha(eids, 102), snap,
                        {101: (1, 0), 102: (2, 1), 103: (0, 0)},
                        [10] * 20, mis_numeros=set(), n_sims=50) is None


def test_correr_fecha_excluye_nuestras_participaciones():
    """La Q y los γ modelan a los RIVALES: contarnos sesgaría justo los marcadores
    diferenciados (mismo criterio que empirical_counts)."""
    from src.clausura.pool_pit import correr_fecha

    eids = [101, 102]
    snap = _snapshot(eids, n_rivales=30)
    mis = {500, 501, 502}                      # 3 de las 30 son nuestras
    pit = correr_fecha(1, _cfg_fecha(eids, 101), snap, {101: (1, 0), 102: (2, 1)},
                       [9] * 27, mis_numeros=mis, n_sims=60)
    assert pit is not None and pit.n_rivales == 27


def test_gammas_fitean_fuera_de_la_fecha_testeada():
    """Fitear γ con los mismos picks que después se testean sería darle al modelo la
    respuesta: se usan solo los partidos de OTRAS fechas."""
    from src.clausura.pool_pit import gammas_y_show

    eids = [101, 102]
    snap = _snapshot(eids, n_rivales=12)
    qs = [_q_plano(), _q_plano()]
    gammas, p_show = gammas_y_show(snap, eids, qs, mis_numeros=set())
    assert len(gammas) == 12 and len(p_show) == 12
    assert (p_show == 1.0).all()               # todos cargaron los 2 partidos
    assert (gammas > 0).all()

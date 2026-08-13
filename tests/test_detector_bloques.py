"""Tests del detector de bloques de compra (scripts/detector_bloques).

Lo que hay que probar es que DETECTA: un bloque optimizador sembrado (raro-pero-
bueno, diverso internamente) tiene que salir arriba, un bloque de clones tiene que
quedar marcado como clon (otra especie), y los fans al azar tienen que quedar en el
medio. Un detector que puntúa alto a cualquiera no detecta nada.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.detector_bloques import (
    Bloque,
    conteos_totales,
    features_bloque,
    fechas_observadas,
    formatear,
    resumen_timestamps,
    score_bloques,
    segmentar,
)
from src.clausura.economics import N_SCORES, score_index

EVENTOS = [101, 102, 103, 104, 105, 106, 107, 108]
COMUNES = [(1, 0), (2, 1), (1, 1), (0, 1), (2, 0)]
RAROS_BUENOS = [(0, 0), (0, 2)]        # lo que el pool evita pero tiene valor
RAROS_TONTOS = [(5, 0), (4, 3)]        # rareza sin valor


def _fila(pid: int, picks: dict[int, tuple[int, int]], ts=None) -> dict:
    return {"participacion_id": pid, "numero": pid + 100_000,
            "picks": {str(k): list(v) for k, v in picks.items()},
            **({"picks_ts": ts} if ts else {})}


def _pool_sintetico(seed: int = 5):
    """200 fans sueltos + un bloque CLON + un bloque OPTIMIZADOR (con huecos de id
    como el nuestro real) + un bloque de fans consecutivos (control)."""
    rng = np.random.default_rng(seed)
    pool = []

    # fans sueltos: ids espaciados (gap 10 ⇒ bloques de tamaño 1), picks chalk
    for i in range(200):
        picks = {e: COMUNES[rng.integers(len(COMUNES))] for e in EVENTOS}
        pool.append(_fila(1_000 + 10 * i, picks))

    # bloque de fans CONSECUTIVOS (control): 6 amigos, chalk independiente
    for j in range(6):
        picks = {e: COMUNES[rng.integers(len(COMUNES))] for e in EVENTOS}
        pool.append(_fila(5_000 + j, picks))

    # bloque CLON: 6 boletas casi idénticas
    base = {e: COMUNES[rng.integers(len(COMUNES))] for e in EVENTOS}
    for j in range(6):
        pool.append(_fila(6_000 + j, dict(base)))

    # bloque OPTIMIZADOR: 8 miembros, ids con huecos ≤6 (como el nuestro), cada uno
    # cubre celdas raras-BUENAS distintas (anti-correlación interna deliberada)
    ids_opt = [7_000, 7_002, 7_003, 7_008, 7_010, 7_014, 7_016, 7_019]
    for j, pid in enumerate(ids_opt):
        picks = {}
        for k, e in enumerate(EVENTOS):
            if (k + j) % 3 == 0:
                picks[e] = RAROS_BUENOS[(k + j) % len(RAROS_BUENOS)]
            else:
                picks[e] = COMUNES[(k + 2 * j) % len(COMUNES)]
        pool.append(_fila(pid, picks))
    return pool, ids_opt


def _grids_eptos():
    """E[pts] sintético: los raros-buenos valen, los raros-tontos no."""
    e = np.full(N_SCORES, 2.0)
    for s in RAROS_BUENOS:
        e[score_index(*s)] = 3.2
    for s in RAROS_TONTOS:
        e[score_index(*s)] = 0.6
    return {eid: e for eid in EVENTOS}


# -------------------- segmentación --------------------

def test_segmenta_con_tolerancia_de_huecos():
    """Nuestra compra real quedó con huecos de hasta 6 ids (globales entre pencas):
    el bloque tiene que sobrevivirlos, y cortarse donde el hueco es mayor."""
    pool = [_fila(p, {101: (1, 0)}) for p in (100, 102, 106, 112, 130, 131)]
    bloques = segmentar(pool, tolerancia=6)
    assert [b.tam for b in bloques] == [4, 2]           # 100-112 junto; 130-131 aparte


def test_marca_nuestro_bloque_por_numero():
    """El nuestro se talla exacto y primero; el vecino pegado (id 100) queda AFUERA
    en su propio bloque en vez de fusionarse con nosotros."""
    pool = [_fila(p, {101: (1, 0)}) for p in (100, 101, 500)]
    bloques = segmentar(pool, tolerancia=6, mis_numeros={100_101})
    assert [b.es_nuestro for b in bloques] == [True, False, False]
    assert bloques[0].ids == [101]


# -------------------- leave-block-out --------------------

def test_la_rareza_no_se_autoexplica():
    """Un bloque grande que juega una celda vacía del pool tiene que verla RARA
    aunque sus propios picks la vuelvan frecuente en los conteos totales."""
    raro = (0, 3)
    bloque = [_fila(10 + j, {101: raro}) for j in range(8)]
    fans = [_fila(1_000 + 10 * i, {101: COMUNES[i % len(COMUNES)]}) for i in range(20)]
    counts = conteos_totales(fans + bloque)

    f_con_lbo = features_bloque(bloque, counts)
    # sin leave-block-out, la frecuencia del pick sería 8/28 ≈ 29% (nada raro);
    # con él, 0 observados fuera del bloque ⇒ rareza alta
    assert f_con_lbo["rareza"] > -np.log10(8 / 28) + 0.5


# -------------------- detección --------------------

@pytest.fixture(scope="module")
def scored():
    pool, ids_opt = _pool_sintetico()
    bloques = segmentar(pool, tolerancia=6)
    return score_bloques(bloques, pool, grids_eptos=_grids_eptos(),
                         b_nula=200, seed=11), ids_opt


def test_el_optimizador_sembrado_sale_primero(scored):
    bloques, ids_opt = scored
    assert min(bloques[0].ids) == ids_opt[0], (
        "el bloque optimizador no salió #1: el detector no detecta lo que dice detectar")
    assert bloques[0].score >= 90


def test_el_clon_queda_marcado_como_clon_no_como_optimizador(scored):
    bloques, _ = scored
    clon = next(b for b in bloques if 6_000 in b.ids)
    assert clon.es_clon
    assert clon.features["similitud"] >= 0.99          # boletas idénticas


def test_los_fans_consecutivos_no_disparan(scored):
    """El control que hace informativo al positivo: 6 amigos con picks chalk
    independientes NO pueden tener firma de optimizador."""
    bloques, _ = scored
    control = next(b for b in bloques if 5_000 in b.ids)
    assert control.score < 85
    assert not control.es_clon


def test_direccion_de_las_features_del_optimizador(scored):
    bloques, ids_opt = scored
    opt = bloques[0]
    assert opt.percentiles["rareza"] > 80          # juega raro…
    assert opt.percentiles["similitud"] < 40       # …sin copiarse a sí mismo…
    assert opt.percentiles["calidad_raros"] > 60   # …y el raro que juega VALE


# -------------------- reporte --------------------

def test_reporte_se_niega_a_concluir_con_una_fecha(scored):
    bloques, _ = scored
    txt = formatear(bloques, n_fechas=1.0)
    assert "INSUFICIENTE" in txt
    assert "optimizador?" not in txt or "3+" in txt


def test_reporte_concluye_con_tres_fechas(scored):
    bloques, _ = scored
    txt = formatear(bloques, n_fechas=3.0)
    assert "firma de optimizador" in txt


def test_calibracion_fallida_grita():
    """Si nuestro bloque no aparece, el resto del reporte no es confiable."""
    pool, _ = _pool_sintetico()
    bloques = segmentar(pool, tolerancia=6)          # sin mis_numeros
    scored_sin = score_bloques(bloques, pool, b_nula=50, seed=3)
    assert "CALIBRACIÓN FALLIDA" in formatear(scored_sin, n_fechas=3.0)


def test_fechas_observadas():
    pool = [_fila(1, {e: (1, 0) for e in EVENTOS})]
    assert fechas_observadas(pool) == 1.0


def test_resumen_timestamps():
    ts = {"101": ["07-08-2026 10:00:00", "07-08-2026 10:00:00"],
          "102": ["07-08-2026 10:03:00", "07-08-2026 18:30:00"]}   # una edición
    b = Bloque(ids=[1], numeros=[2], filas=[_fila(1, {101: (1, 0)}, ts=ts)])
    r = resumen_timestamps(b)
    assert "3 min" in r and "1 edicion" in r


def test_nuestro_bloque_se_talla_exacto_aunque_haya_vecinos_pegados():
    """Primera corrida real: la tolerancia fusionó nuestras 12 con ~17 vecinos y la
    calibración salió #6. La membresía nuestra es CONOCIDA: se talla por número."""
    pool = [_fila(p, {101: (1, 0)}) for p in (200, 201, 203, 205, 206, 208)]
    mis = {100_201, 100_205}                       # numero = id + 100000
    bloques = segmentar(pool, tolerancia=6, mis_numeros=mis)
    nuestro = next(b for b in bloques if b.es_nuestro)
    assert sorted(nuestro.ids) == [201, 205]       # exacto, sin vecinos
    assert not any(i in (201, 205) for b in bloques if not b.es_nuestro for i in b.ids)


def test_mega_cadenas_quedan_fuera_del_scoring():
    """La ola de compras pre-kickoff encadena decenas de compradores (178 en el v9
    real): es una ventana masiva, no una persona — no puede puntuar."""
    pool, _ = _pool_sintetico()
    pool += [_fila(9_000 + 2 * i, {e: COMUNES[i % len(COMUNES)] for e in EVENTOS})
             for i in range(60)]                    # cadena de 60 con gaps de 2
    bloques = segmentar(pool, tolerancia=6)
    scored = score_bloques(bloques, pool, b_nula=50, seed=7, max_tam=40)
    assert all(b.tam <= 40 for b in scored)


def test_un_bloque_estilo_NUESTRO_dispara_el_perfil_ev():
    """La calibración real del 13/8 refutó el perfil único: nuestro sistema no juega
    raro — juega CALIDAD máxima repartida entre las mejores celdas. Un bloque con
    esa huella (todos los picks en el top de E[pts], miembros no idénticos) tiene
    que puntuar alto por el perfil EV aunque su rareza sea mediocre."""
    pool, _ = _pool_sintetico()
    top_ev = [(1, 0), (1, 1)]                     # las 2 celdas de mayor E[pts]…
    grids = _grids_eptos()
    for eid in EVENTOS:
        grids[eid] = grids[eid].copy()
        for s in top_ev:
            grids[eid][score_index(*s)] = 3.5     # …bien separadas del resto
    rng = np.random.default_rng(29)
    ids_ev = list(range(8_000, 8_012))
    for pid in ids_ev:
        picks = {e: top_ev[rng.integers(2)] for e in EVENTOS}
        pool.append(_fila(pid, picks))

    bloques = segmentar(pool, tolerancia=6)
    scored = score_bloques(bloques, pool, grids_eptos=grids, b_nula=200, seed=13)
    ev_block = next(b for b in scored if 8_000 in b.ids)
    assert ev_block.score_ev >= 85
    assert ev_block.score_ev > ev_block.score_raro     # y el perfil correcto gana
    assert not ev_block.es_clon                        # repartirse ≠ copiarse

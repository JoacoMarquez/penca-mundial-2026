"""Modo offline (src.clausura.offline): Supermatch bloquea el API desde el 5/10."""

import json

from src.clausura.odds import EventOdds
from src.clausura.offline import (
    cuotas_manuales,
    load_offline,
    merge_cuotas,
    ranking_manual,
    resultados_offline,
)


def test_resultados_de_postmortems_y_manuales_ganan(tmp_path):
    (tmp_path / "fecha_08.json").write_text(json.dumps({"resultados": {"2140": [1, 0]}}))
    (tmp_path / "fecha_09.json").write_text(
        json.dumps({"resultados": {"2144": [3, 2], "2147": [9, 9]}}))
    off = {"resultados": {2147: [0, 2], 2150: [2, 0]}}
    res = resultados_offline(off, pm_dir=tmp_path)
    assert res == {2140: (1, 0), 2144: (3, 2), 2147: (0, 2), 2150: (2, 0)}


def test_sin_archivo_es_vacio(tmp_path):
    assert load_offline(tmp_path / "no.yaml") == {}
    assert resultados_offline({}, pm_dir=tmp_path / "nada") == {}


def test_ranking_manual_a_enteros():
    assert ranking_manual({"ranking": {"899258512": "221"}}) == {899258512: 221}


def test_cuotas_manuales_pisan_al_cache_por_nombre():
    off = {"cuotas": [{"local": "Peñarol", "visitante": "Progreso",
                       "1": 1.45, "X": 4.2, "2": 7.0,
                       "over_2_5": 1.9, "under_2_5": 1.85}]}
    man = cuotas_manuales(off)
    assert man[0].x1x2 == {"home": 1.45, "draw": 4.2, "away": 7.0}
    assert man[0].totals == {"2.5": {"over": 1.9, "under": 1.85}}
    cache = [EventOdds("sm:1", "Penarol", "Progreso", "", "", x1x2={"home": 2.0}),
             EventOdds("sm:2", "Nacional", "Cerro", "", "", x1x2={"home": 1.5})]
    merged = merge_cuotas(cache, man)
    assert [e.event_id for e in merged] == ["sm:2", man[0].event_id]


def test_liquidar_en_snapshot_suma_los_puntos_del_partido():
    from src.clausura.offline import liquidar_en_snapshot
    from src.clausura.scoring import supermatch_points
    snap = {"participaciones": [
        {"numero": 1, "puntos": 100, "picks": {"2147": [0, 2], "2149": [2, 1]}},
        {"numero": 2, "puntos": 90, "picks": {"2147": [1, 1]}},
        {"numero": 3, "puntos": 80, "picks": {}},
    ]}
    n = liquidar_en_snapshot(snap, [2147, 2149], {2147: (0, 2), 2149: (2, 1)}, {2149})
    p = snap["participaciones"]
    assert p[0]["puntos"] == 100 + supermatch_points((0, 2), (0, 2)) \
        + supermatch_points((2, 1), (2, 1), True)
    assert p[1]["puntos"] == 90 + supermatch_points((1, 1), (0, 2))
    assert p[2]["puntos"] == 80
    assert n == (2 if supermatch_points((1, 1), (0, 2)) else 1)


def test_aviso_cuotas_cache_offline_no_promete_rerun():
    from src.clausura.picks import aviso_cuotas_cache
    aviso, consejo = aviso_cuotas_cache(27.4, offline=True, n_manuales=0)
    assert "27.4h" in aviso and "rerun que corrija" in consejo
    assert "va a proponer" not in consejo
    aviso, _ = aviso_cuotas_cache(80.0, offline=True, n_manuales=3)
    assert "3 partidos con cuota copiada a mano" in aviso
    _, consejo = aviso_cuotas_cache(5.0, offline=False, n_manuales=0)
    assert "rerun con ES vivo" in consejo

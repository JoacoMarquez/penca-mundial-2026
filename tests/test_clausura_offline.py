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

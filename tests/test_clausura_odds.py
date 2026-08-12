"""Tests de parseo de odds de Supermatch ES para la Primera uruguaya.

Fixture recortada del evento real Peñarol vs Wanderers (capturado 2026-08-04).
"""

from src.clausura.odds import parse_event

HIT = {
    "_id": "71771860",
    "_source": {
        "description": "Peñarol vs Wanderers",
        "leagueName": "Uruguay",
        "championshipName": "Primera División",
        "dateTime": 1754424000000,
        "betLines": [
            {"type": "3w", "description": "1x2", "options": [
                {"result": "Peñarol", "dividend": 1.53},
                {"result": "empate", "dividend": 3.8},
                {"result": "Wanderers", "dividend": 5.2},
            ]},
            {"type": "to", "description": "Total ( 2.5 )", "options": [
                {"result": "más de 2.5", "dividend": 2.0},
                {"result": "menos de 2.5", "dividend": 1.53},
            ]},
            # total de córners: mismo type "to", NO debe entrar
            {"type": "to", "description": "Total de córners ( 9.5 )", "options": [
                {"result": "más de 9.5", "dividend": 1.9},
                {"result": "menos de 9.5", "dividend": 1.8},
            ]},
            {"type": "ftnw", "description": "Marcador exacto", "options": [
                {"result": "1:0", "dividend": 4.85},
                {"result": "2:0", "dividend": 5.2},
                {"result": "2:1", "dividend": 7.0},
                {"result": "0:0", "dividend": 8.0},
                {"result": "otro", "dividend": 23.0},
            ]},
            {"type": "ftnw", "description": "Peñarol Goles exactos", "options": [
                {"result": "0", "dividend": 4.25},
                {"result": "1", "dividend": 2.65},
                {"result": "2", "dividend": 3.2},
                {"result": "3+", "dividend": 3.6},
            ]},
            {"type": "ftnw", "description": "Wanderers Goles exactos", "options": [
                {"result": "0", "dividend": 1.75},
                {"result": "1", "dividend": 2.38},
                {"result": "2", "dividend": 6.2},
                {"result": "3+", "dividend": 21.0},
            ]},
            {"type": "ftnw", "description": "Margen de victoria", "options": [
                {"result": "empate", "dividend": 3.8},
                {"result": "Peñarol por 1", "dividend": 3.05},
                {"result": "Peñarol por 2", "dividend": 4.05},
                {"result": "Peñarol por 3+", "dividend": 5.1},
                {"result": "Wanderers por 1", "dividend": 6.2},
                {"result": "Wanderers por 2", "dividend": 19.0},
                {"result": "Wanderers por 3+", "dividend": 67},
            ]},
            # mercado de 1º tiempo con nombre parecido: NO debe entrar
            {"type": "ftnw", "description": "1º Mitad - marcador exacto", "options": [
                {"result": "1:0", "dividend": 3.15},
            ]},
        ],
    },
}


def test_parse_event_basico():
    ev = parse_event(HIT)
    assert ev is not None
    assert ev.home == "Peñarol" and ev.away == "Wanderers"
    assert ev.x1x2 == {"home": 1.53, "draw": 3.8, "away": 5.2}


def test_totales_solo_de_goles():
    ev = parse_event(HIT)
    assert set(ev.totals) == {"2.5"}
    assert ev.totals["2.5"] == {"over": 2.0, "under": 1.53}


def test_marcador_exacto_del_partido_completo():
    ev = parse_event(HIT)
    assert ev.correct_score["2:0"] == 5.2
    assert ev.correct_score["otro"] == 23.0
    # el de 1º mitad no contamina
    assert ev.correct_score["1:0"] == 4.85


def test_goles_por_equipo():
    ev = parse_event(HIT)
    assert ev.home_goals == {"0": 4.25, "1": 2.65, "2": 3.2, "3+": 3.6}
    assert ev.away_goals["3+"] == 21.0


def test_margen_de_victoria_mapeado():
    ev = parse_event(HIT)
    assert ev.margin["D"] == 3.8
    assert ev.margin["H+2"] == 4.05
    assert ev.margin["A+3"] == 67


def test_evento_sin_vs_se_descarta():
    assert parse_event({"_id": "x", "_source": {"description": "Especial", "dateTime": 1}}) is None


# -------------------- persistencia --------------------

def _ev_odds(home="Peñarol", away="Wanderers", start="2099-01-01T00:00:00+00:00",
             con_1x2=True):
    from src.clausura.odds import EventOdds
    return EventOdds(event_id=f"sm:{home}-{away}", home=home, away=away,
                     start_utc=start, fetched_utc="x",
                     x1x2={"home": 2.0, "draw": 3.2, "away": 3.8} if con_1x2 else {})


def test_odds_roundtrip_y_edad(tmp_path, monkeypatch):
    """Un outage del ES degradaba todo a ratings EN SILENCIO: el cache versionado
    devuelve las cuotas de la corrida anterior con su edad, y vence a las 24h."""
    import src.clausura.odds as odds_mod
    from src.clausura.odds import load_cached_odds, save_odds_snapshot
    monkeypatch.setattr(odds_mod, "ODDS_DIR", tmp_path)

    assert load_cached_odds() is None                       # sin cache: None
    save_odds_snapshot([_ev_odds()])
    cargado = load_cached_odds()
    assert cargado is not None
    odds, edad_h = cargado
    assert odds[0].x1x2["home"] == 2.0 and edad_h < 0.1

    # snapshot viejo: no se usa (la cuota de hace 3 días ya se movió con noticias)
    viejo = tmp_path / "odds_20200101T000000Z.json"
    for f in tmp_path.glob("odds_*.json"):
        f.rename(viejo)
    assert load_cached_odds(max_age_h=24.0) is None

    # lista vacía no pisa el cache con un archivo inútil
    save_odds_snapshot([])
    assert sorted(tmp_path.glob("odds_*.json")) == [viejo]


def test_mercados_perdidos_solo_futuros_con_1x2():
    """'Lo tenía y lo perdió' huele a rename del keyword; 'nunca lo tuvo' o 'ya
    empezó' es lo normal del ES que publica solo la fecha próxima."""
    from src.clausura.odds import mercados_perdidos

    previos = [
        _ev_odds("Peñarol", "Wanderers"),                          # futuro con 1x2
        _ev_odds("Cerro", "Racing", start="2020-01-01T00:00:00+00:00"),  # ya empezó
        _ev_odds("Danubio", "Albion", con_1x2=False),              # nunca tuvo 1x2
    ]
    actuales = [_ev_odds("Nacional", "Liverpool")]
    assert mercados_perdidos(previos, actuales) == ["Peñarol vs Wanderers"]
    # si sigue presente, no es pérdida
    assert mercados_perdidos(previos, previos[:1]) == []

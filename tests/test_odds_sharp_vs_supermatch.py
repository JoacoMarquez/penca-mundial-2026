"""Tests del comparador de cuotas sharp vs Supermatch (scripts/odds_sharp_vs_supermatch).

Lo que hay que blindar es el MATCHEO: un partido emparejado mal produce un "sesgo"
que es puro error de matcheo. Es el mismo modo de falla que el cruce Cerro/Cerro
Largo de agosto, que envenenó grillas durante días sin que nada sonara.
"""

from __future__ import annotations

from scripts.odds_sharp_vs_supermatch import (
    ParMatcheado,
    comparar_par,
    emparejar,
    resumir,
)


def _ev(home, away, hora="2026-08-15T20:45:00+00:00", x1x2=None):
    return {"home": home, "away": away, "start_utc": hora,
            "x1x2": x1x2 or {"home": 2.0, "draw": 3.2, "away": 3.8}}


# -------------------- matcheo --------------------

def test_matchea_nombres_localizados():
    """Supermatch escribe en español y con sufijos; Pinnacle en inglés."""
    sm = [_ev("Defensor Sporting", "Liverpool (URU)")]
    pin = [_ev("Defensor", "Liverpool Montevideo")]
    assert len(emparejar(sm, pin)) == 1


def test_no_matchea_cerro_con_cerro_largo():
    """EL caso: "Cerro" es substring de "Cerro Largo". Un match acá compara dos
    partidos distintos y todo el sesgo medido después es basura."""
    sm = [_ev("Cerro", "Albion")]
    pin = [_ev("Cerro Largo", "Boston River")]
    assert emparejar(sm, pin) == []


def test_no_matchea_partidos_de_otro_dia():
    """Mismos equipos, otra fecha: es el partido de la rueda siguiente."""
    sm = [_ev("Peñarol", "Nacional", "2026-08-15T20:45:00+00:00")]
    pin = [_ev("Penarol", "Nacional", "2026-10-20T20:45:00+00:00")]
    assert emparejar(sm, pin) == []


def test_matchea_con_kickoff_corrido_unas_horas():
    """Los books no siempre coinciden al minuto en el horario publicado."""
    sm = [_ev("Wanderers", "Cerro Largo", "2026-08-15T20:45:00+00:00")]
    pin = [_ev("Montevideo Wanderers", "Cerro Largo", "2026-08-15T23:00:00+00:00")]
    assert len(emparejar(sm, pin)) == 1


def test_sin_candidato_no_inventa():
    sm = [_ev("Progreso", "Deportivo Maldonado")]
    assert emparejar(sm, []) == []


# -------------------- alineación por rol --------------------

def test_alinea_por_ROL_no_por_localia():
    """El sesgo favorito-longshot es sobre el ROL. Si el favorito es el visitante,
    comparar 'home vs home' mezclaría favorito con no-favorito entre partidos."""
    sm = _ev("Cerro", "Albion", x1x2={"home": 4.0, "draw": 3.4, "away": 2.1})
    pin = _ev("Cerro", "Albion", x1x2={"home": 4.2, "draw": 3.5, "away": 2.0})
    par = comparar_par(sm, pin)
    assert par.favorito == "away"                 # Albion es favorito en Pinnacle
    assert par.sm_fav > par.sm_dog                # y el fav de SM es el mismo lado


def test_vig_se_reporta_de_las_odds_crudas():
    sm = _ev("A", "B", x1x2={"home": 2.0, "draw": 3.0, "away": 4.0})     # ~1.083
    pin = _ev("A", "B", x1x2={"home": 2.05, "draw": 3.4, "away": 4.2})   # más chico
    par = comparar_par(sm, pin)
    assert par.vig_supermatch > par.vig_pinnacle > 0


def test_diferencias_suman_cero():
    """Las dos distribuciones están normalizadas: las tres diferencias suman 0."""
    par = comparar_par(_ev("A", "B"), _ev("A", "B", x1x2={"home": 2.1, "draw": 3.3,
                                                          "away": 3.6}))
    assert abs(par.dif_fav + par.dif_empate + par.dif_dog) < 1e-9


# -------------------- reporte --------------------

def _par(dif_fav: float, i: int = 0) -> ParMatcheado:
    """Par sintético con una diferencia dada en el favorito."""
    return ParMatcheado(
        partido=f"A{i} vs B{i}", start_utc="2026-08-15T20:45:00+00:00",
        vig_supermatch=0.15, vig_pinnacle=0.025,
        sm_fav=0.50 + dif_fav, pin_fav=0.50,
        sm_empate=0.28, pin_empate=0.28,
        sm_dog=0.22 - dif_fav, pin_dog=0.22, favorito="home")


def test_reporte_marca_sesgo_estable_y_su_direccion():
    pares = [_par(0.030 + 0.001 * i, i) for i in range(8)]     # consistente y grande
    txt = resumir(pares)
    assert "Sesgo estable" in txt and "SOBREVALÚA" in txt
    assert "8/8 positivos" in txt


def test_reporte_no_concluye_con_ruido():
    pares = [_par(0.03 if i % 2 else -0.03, i) for i in range(8)]   # media ~0
    txt = resumir(pares)
    assert "Sin sesgo distinguible" in txt


def test_reporte_no_concluye_con_pocos_partidos():
    """Una fecha son 8 partidos: con menos no se puede separar sesgo de ruido."""
    txt = resumir([_par(0.05, i) for i in range(4)])
    assert "Sin sesgo distinguible" in txt


def test_reporte_detecta_el_signo_contrario():
    txt = resumir([_par(-0.030 - 0.001 * i, i) for i in range(8)])
    assert "Sesgo estable" in txt and "SUBVALÚA" in txt


def test_reporte_vacio_no_rompe():
    assert "sin partidos" in resumir([])


# -------------------- el caso adversarial de Cerro/Cerro Largo --------------------

def test_no_matchea_cerro_con_cerro_largo_NI_con_el_mismo_rival():
    """La versión filosa: si los dos jugaran contra el mismo rival a la misma hora,
    un matcher por substring o por subconjunto los da por iguales.

    `_norm("Cerro") in _norm("Cerro Largo")` es exactamente el bug que envenenó
    grillas en agosto. Acá el token "largo" no está en la lista de ruido, así que
    discrimina.
    """
    sm = [_ev("Cerro", "Albion")]
    pin = [_ev("Cerro Largo", "Albion")]
    assert emparejar(sm, pin) == []


def test_matchea_igual_con_ruido_de_ciudad_y_pais():
    """Lo que sobra de los dos lados es ruido conocido: sí es el mismo equipo."""
    sm = [_ev("Liverpool (URU)", "Defensor Sporting")]
    pin = [_ev("Liverpool Montevideo", "Defensor")]
    assert len(emparejar(sm, pin)) == 1


def test_elige_el_correcto_entre_cerro_y_cerro_largo():
    """Con los dos candidatos presentes tiene que quedarse con el que corresponde."""
    sm = [_ev("Cerro Largo", "Albion")]
    pin = [_ev("Cerro", "Albion"), _ev("Cerro Largo", "Albion")]
    pares = emparejar(sm, pin)
    assert len(pares) == 1 and pares[0][1]["home"] == "Cerro Largo"

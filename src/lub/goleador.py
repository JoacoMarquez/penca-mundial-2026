"""Goleador uruguayo de la LUB: P(goleador) desde estadísticas reales, CONJUNTA con la temporada.

Reglamento (penca 48, Art. 6): 25 puntos al jugador de nacionalidad uruguaya "que más
puntos convierta en la totalidad del torneo" — puntos TOTALES, playoffs incluidos. O
sea que no gana el de mejor promedio sino promedio × partidos jugados, y los partidos
los define hasta dónde llega su equipo: el goleador está correlacionado con el campeón
(en la 25/26 ganó Vescovi, 612 pts en 39 partidos con Peñarol campeón; Pomoli, su
compañero, 552). Por eso se simula sobre los MISMOS sorteos de la temporada
(`Sorteo.partidos`) y no como un prior independiente.

Modelo por jugador i y sorteo s:
    ppg[s,i]   ~ N(a + b·ppg_prev_i, σ_i)            (persistencia medida 24/25 → 25/26)
    pj[s,i]    ~ Binomial(partidos del equipo[s], disp[s,i])
    total[s,i] = ppg·pj + N(0, SD_PARTIDO·√pj)
    goleador[s] = argmax_i total[s,i]

Datos: Genius Sports (estadísticas oficiales de la FUBB, puntos y partidos por jugador
y temporada) + planteles 26/27 (fichas mayores y sub 23 = nacionales). Los que no
tienen temporada previa en la LUB entran con un prior MANUAL (`PRIORES_MANUALES`) o con
el genérico de recién llegado, y eso es lo más frágil del modelo: Vescovi llegó de
afuera en la 25/26 y la estadística de la temporada anterior no lo habría visto.

Uso:
    python -m src.lub.goleador --bajar      # scrapea Genius y arma data/lub/goleador_candidatos.json
    python -m src.lub.goleador              # P(goleador) con la temporada simulada actual
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.lub.data import DATA_DIR

log = logging.getLogger(__name__)

GENIUS = "https://hosted.dcd.shared.geniussports.com/FUBB/es/competition"
COMPETENCIAS = {"24/25": 39840, "25/26": 42104, "amistosos_2026": 50123}
GENIUS_JSON = DATA_DIR / "genius_jugadores.json"
PLANTELES_JSON = DATA_DIR / "planteles_2627.json"
CANDIDATOS_JSON = DATA_DIR / "goleador_candidatos.json"

# Genius → nombre del penca-api
EQUIPO_PENCA = {
    "AGUADA": "Aguada", "ATENAS": "Atenas", "BIGUA": "Bigua", "CORDON": "Cordón",
    "D SPORTING": "Defensor Sporting", "GOES": "Goes", "HEBRAICA Y MACABI": "Hebraica Macabi",
    "LARRAÑAGA": "Larrañaga", "MALVIN": "Malvín", "NACIONAL": "Nacional", "PEÑAROL": "Peñarol",
    "UNION ATLETICA": "Unión Atlética", "URUNDAY UNIVERSITARIO": "Urunday Universitario",
    "WELCOME": "Welcome", "TROUVILLE": "Trouville", "URUPAN": "Urupan",
    "FERRO CARRIL": "Ferro Carril", "REMEROS MERCEDES": "Remeros Mercedes",
}

# Persistencia del PPG entre temporadas, medida el 2/10/2026 sobre 61 jugadores
# nacionales con ≥12 partidos en la 24/25 y en la 25/26: ppg' = 2,95 + 0,57·ppg,
# residuo sd 2,0 si siguió en el mismo club y 2,8 si cambió (rol nuevo).
PPG_A, PPG_B = 2.95, 0.57
SD_MISMO, SD_CAMBIO = 2.0, 2.8
SD_SIN_TEMPORADA = 1.0          # extra si la última temporada en la LUB es la 24/25
# Disponibilidad (partidos jugados / partidos del equipo): titulares nacionales de la
# 25/26 con ≥7 ppg → media 0,95, p10 0,83. Ese número sale de los que JUGARON, así que
# no ve la lesión que te borra la temporada: se agrega una mezcla con P_LESION.
DISP_A, DISP_B = 18.0, 1.0       # Beta(18, 1): media 0,947
P_LESION, LESION_RANGO = 0.10, (0.2, 0.7)
SD_PARTIDO = 6.0                 # sd de puntos de un partido (ruido de juego a juego)
PPG_NUEVO, SD_NUEVO = 5.0, 3.0   # recién llegado sin datos ni prior manual

# Recién llegados o vueltas con información de prensa (2/10/2026). Fuente y criterio
# en cada uno; mu/sd en puntos por partido, disp = disponibilidad media esperada.
PRIORES_MANUALES = {
    # 34 años, volvió del retiro tras romperse el tendón de Aquiles a mitad de 2025
    # (montevideo.com.uy). Fue figura de la selección; rol y minutos inciertos.
    "Mathías Calfani": {"mu": 8.0, "sd": 3.5, "disp": 0.75},
    # nacido en Curitiba, viene del Cearense (Brasil), ficha mayor en Malvín. Sin datos LUB.
    "Fernando Salsamendi": {"mu": 7.0, "sd": 3.5, "disp": 0.9},
}


# Menú de Supermatch → nombre en Genius/planteles, cuando difieren más que en tildes.
ALIAS_MENU = {
    "Ignaxio Xavier": "Ignacio Xavier",
    "Sebastian Otonello": "Sebastián Ottonello",
    "Juan Ducasse": "Juan Ignacio Ducasse",
}


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper().strip()


# ---------------------------------------------------------------- datos

def bajar_genius() -> dict:
    """Puntos y partidos por jugador y club en cada competencia de COMPETENCIAS."""
    import httpx
    res: dict[str, list[dict]] = {}
    with httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=30, follow_redirects=True) as c:
        for temp, comp in COMPETENCIAS.items():
            h = c.get(f"{GENIUS}/{comp}/teams").text
            filas = []
            # cada club aparece dos veces (logo sin texto + nombre): solo los que tienen nombre
            for tid, team in sorted({(t, n) for t, n in re.findall(r'team/(\d+)\?">([^<]*)<', h)
                                     if n.strip()}):
                t = c.get(f"{GENIUS}/{comp}/team/{tid}/statistics").text
                i = t.find("team-stats")
                t = t[i:t.find("</table>", i)]
                for pid, name, rest in re.findall(r'person/(\d+)\?">([^<]+)</a>\s*</td>(.*?)</tr>',
                                                  t, flags=re.S):
                    tds = [x.strip().replace(",", "") for x in re.findall(r"<td[^>]*>([^<]*)</td>", rest)]
                    if len(tds) >= 4:
                        filas.append({"pid": int(pid), "nombre": name.strip(), "equipo": team.strip(),
                                      "pj": int(float(tds[0])), "pts": int(float(tds[3]))})
                time.sleep(0.5)
            res[temp] = filas
            log.info("Genius %s: %d filas jugador-club", temp, len(filas))
    return res


def _clave(nombre: str) -> tuple[str, str]:
    p = _norm(nombre).split()
    return p[0][0], " ".join(p[1:])


def _buscar(genius: dict, nombre: str) -> list[dict]:
    """Filas de Genius del jugador ("S. VESCOVI"). Desambigua homónimos por partidos."""
    ini, ape = _clave(nombre)
    filas = []
    for temp, fs in genius.items():
        for f in fs:
            if not f.get("equipo"):      # filas duplicadas sin club de bajadas viejas
                continue
            n = _norm(f["nombre"])
            if not n.startswith(ini + "."):
                continue
            ap = n.split(". ", 1)[1] if ". " in n else ""
            # "K. WACHSMAN" vs "Kiril Wachsmann": prefijo de 6 letras del apellido
            if ap == ape or (len(ape) >= 6 and ap[:6] == ape[:6] and ap.split()[-1][:4] == ape.split()[-1][:4]):
                filas.append({**f, "temporada": temp})
    if not filas and len(nombre.split()) >= 3:
        # nombre compuesto: "Juan Ignacio Ducasse" figura en Genius como "J. DUCASSE"
        partes = nombre.split()
        return _buscar(genius, f"{partes[0]} {partes[-1]}")
    pids = {f["pid"] for f in filas}
    if len(pids) > 1:   # homónimos: el de más partidos en las dos temporadas
        pj = {p: sum(f["pj"] for f in filas if f["pid"] == p and f["temporada"] != "amistosos_2026")
              for p in pids}
        keep = max(pj, key=pj.get)
        log.info("homónimos para %s: %s → me quedo con %d", nombre, sorted(pids), keep)
        filas = [f for f in filas if f["pid"] == keep]
    return filas


def armar_candidatos(genius: dict, planteles: dict) -> list[dict]:
    out = []
    for equipo, pl in planteles.items():
        if equipo.startswith("_"):
            continue
        for tipo in ("mayores", "sub23"):
            for nombre in pl[tipo]:
                filas = _buscar(genius, nombre)
                temps = {}
                for t in ("25/26", "24/25"):
                    fs = [f for f in filas if f["temporada"] == t]
                    if fs:
                        pj, pts = sum(f["pj"] for f in fs), sum(f["pts"] for f in fs)
                        clubes = [EQUIPO_PENCA.get(f["equipo"], f["equipo"]) for f in fs]
                        temps[t] = {"pj": pj, "pts": pts, "ppg": round(pts / pj, 2) if pj else 0.0,
                                    "club": max(zip([f["pj"] for f in fs], clubes))[1]}
                out.append({"nombre": nombre, "equipo": equipo, "tipo": tipo,
                            "pid": filas[0]["pid"] if filas else None, **temps})
    return out


# ---------------------------------------------------------------- modelo

@dataclass
class Candidato:
    nombre: str
    equipo: str
    mu: float          # PPG esperado
    sd: float
    disp: float        # disponibilidad media (para la Beta)
    origen: str


def priors(candidatos: list[dict], min_pj: int = 8) -> list[Candidato]:
    """PPG esperado y su incertidumbre por candidato (ver la docstring del módulo)."""
    out = []
    for c in candidatos:
        m = PRIORES_MANUALES.get(c["nombre"])
        if m:
            out.append(Candidato(c["nombre"], c["equipo"], m["mu"], m["sd"], m["disp"], "manual"))
            continue
        prev = next((c.get(t) for t in ("25/26", "24/25") if c.get(t) and c[t]["pj"] >= min_pj), None)
        if prev is None:
            out.append(Candidato(c["nombre"], c["equipo"], PPG_NUEVO, SD_NUEVO, DISP_A / (DISP_A + DISP_B),
                                 "sin datos"))
            continue
        cambio = prev["club"] != c["equipo"]
        sd = SD_CAMBIO if cambio else SD_MISMO
        if c.get("25/26") is not prev:
            sd = float(np.hypot(sd, SD_SIN_TEMPORADA))
        out.append(Candidato(c["nombre"], c["equipo"], PPG_A + PPG_B * prev["ppg"], sd,
                             DISP_A / (DISP_A + DISP_B),
                             f"{'25/26' if c.get('25/26') is prev else '24/25'} {prev['ppg']:.1f} ppg"
                             f"{' (cambió de club)' if cambio else ''}"))
    return out


def simular_goleador(partidos_equipo: np.ndarray, equipos: list[str], cands: list[Candidato],
                     seed: int = 20261012) -> tuple[np.ndarray, np.ndarray]:
    """(goleador por sorteo (S,), totales (S, G)) dados los partidos de cada equipo (S, T).

    Los candidatos de equipos que no están en `equipos` se descartan con un warning."""
    rng = np.random.default_rng(seed)
    S = partidos_equipo.shape[0]
    ix = {e: i for i, e in enumerate(equipos)}
    faltan = sorted({c.equipo for c in cands} - set(ix))
    if faltan:
        raise ValueError(f"equipos de los candidatos que no están en la temporada: {faltan}")
    G = len(cands)
    mu = np.array([c.mu for c in cands]); sd = np.array([c.sd for c in cands])
    ppg = np.clip(mu[None, :] + rng.normal(0, 1, (S, G)) * sd[None, :], 0, None)
    d_med = np.array([c.disp for c in cands])
    # Beta con media d_med y la concentración medida (DISP_A + DISP_B)
    k = DISP_A + DISP_B
    disp = rng.beta(d_med * k, (1 - d_med) * k, (S, G))
    lesion = rng.random((S, G)) < P_LESION
    disp = np.where(lesion, rng.uniform(*LESION_RANGO, (S, G)), disp)
    eq = np.array([ix[c.equipo] for c in cands])
    pj_eq = partidos_equipo[:, eq]                                    # (S, G)
    pj = rng.binomial(pj_eq.astype(np.int64), disp)
    total = ppg * pj + rng.normal(0, 1, (S, G)) * SD_PARTIDO * np.sqrt(pj)
    return total.argmax(axis=1), total


def en_menu(cands: list[Candidato], menu: list[str]) -> dict[int, str]:
    """índice de candidato → nombre tal cual figura en el menú de la web.

    El ganador real se sigue sorteando entre TODOS los candidatos (si gana uno que no
    está en el menú, nadie suma esos 25), pero solo se puede cargar lo que ofrece el menú."""
    clave = {_norm(ALIAS_MENU.get(m, m)): m for m in menu}
    out = {i: clave[_norm(c.nombre)] for i, c in enumerate(cands) if _norm(c.nombre) in clave}
    faltan = set(menu) - set(out.values())
    if faltan:
        log.warning("menú de goleador sin candidato en el modelo (no se asignan): %s", sorted(faltan))
    return out


def cargar_candidatos(path: Path | None = None) -> list[Candidato] | None:
    path = path or CANDIDATOS_JSON
    if not path.exists():
        return None
    return priors(json.loads(path.read_text()))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bajar", action="store_true", help="scrapear Genius y rearmar los candidatos")
    ap.add_argument("--sims", type=int, default=20_000)
    a = ap.parse_args()
    if a.bajar:
        genius = bajar_genius()
        GENIUS_JSON.write_text(json.dumps(genius, ensure_ascii=False, indent=0))
    genius = json.loads(GENIUS_JSON.read_text())
    planteles = json.loads(PLANTELES_JSON.read_text())
    cands = armar_candidatos(genius, planteles)
    CANDIDATOS_JSON.write_text(json.dumps(cands, ensure_ascii=False, indent=1))
    log.info("%d candidatos → %s", len(cands), CANDIDATOS_JSON)

    from datetime import datetime, timezone
    from src.lub.data import load_temporadas
    from src.lub.model import PARAMS_PROD, fit
    from src.lub.picks import TEMPORADA, slots_temporada
    from src.lub.season import Config, simular
    partidos = load_temporadas()
    rt = fit(partidos, datetime.now(timezone.utc), TEMPORADA, PARAMS_PROD)
    so = simular(rt, slots_temporada(partidos), Config(n_sims=a.sims))
    pc = priors(cands)
    gol, total = simular_goleador(so.partidos, so.equipos, pc)
    p = np.bincount(gol, minlength=len(pc)) / len(gol)
    campeon = so.equipos
    print(f"{'jugador':24s} {'equipo':22s} {'P(gol)':>7s} {'E[pts]':>7s}  P(gol | su equipo campeón)  origen")
    for i in np.argsort(-p)[:15]:
        c = pc[i]
        t = campeon.index(c.equipo)
        m = so.campeon == t
        pc_c = (gol[m] == i).mean() if m.any() else float("nan")
        print(f"{c.nombre:24s} {c.equipo:22s} {p[i]:7.1%} {total[:, i].mean():7.0f}  {pc_c:7.1%} "
              f"(P camp {m.mean():.0%})  {c.origen}")


if __name__ == "__main__":
    main()

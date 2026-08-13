"""¿Las cuotas de Supermatch —única fuente del 70% de λ— están sesgadas vs un sharp?

MOTIVACIÓN (auditoría del 13/8, dos agentes por caminos independientes). El modelo
del Clausura toma el 70% de cada λ del Elasticsearch de Supermatch (`MARKET_WEIGHT`),
una casa recreativa con monopolio local. Y este mismo repo tiene la evidencia de que
esas líneas se desvían del precio justo: `src/valuebet/` existía para capturar +EV
CONTRA Supermatch usando Pinnacle como referencia sharp. O sea: le estamos creyendo
al 70% a un termómetro que ya demostramos que marca mal.

Medido en vivo el 13/8 desde el pipeline de producción, sobre los 8 partidos de la
Fecha 2: el **overround del 1X2 de Supermatch va de 10,7% a 15,8%**. Pinnacle corre
~2-3% en fútbol. Con ese vig, dos cosas dejan de ser detalle:

  1. **Qué método de de-vig** se usa. Producción usa `proportional` en el 1X2
     (`picks.market_lambdas`). Contra Shin, la diferencia en el favorito va de
     +0,15 pp en un partido parejo a **+3,5 pp en Peñarol–Central Español** — el
     efecto crece con lo desparejo del partido, que es justo donde el pool
     concentra. En la grilla resultante eso movió el marcador modal en 1 de 8
     partidos (Progreso–Maldonado, 1-1 → 0-1). Real, pero de segundo orden.
  2. **Si además la línea cruda está shadeada.** Eso el de-vig no lo arregla: si la
     plata de hinchas de Peñarol/Nacional achica su cuota, ninguna renormalización
     lo recupera. Es lo que este script mide.

## Qué hace

Compara, partido a partido, el 1X2 de-vigueado de Supermatch contra el de Pinnacle,
y descompone la diferencia por ROL (favorito / empate / no-favorito, según Pinnacle)
en vez de por local/visitante: el sesgo favorito-longshot es una afirmación sobre el
rol, no sobre la localía.

Salida: por partido y agregado. El signo que importa es `dif_fav` medio —
Supermatch − Pinnacle en la probabilidad del favorito:

    dif_fav > 0  ⇒ Supermatch SOBREVALÚA al favorito ⇒ nuestras grillas cargan de
                   más al chalk ⇒ estamos jugando más al favorito de lo que
                   corresponde, en la dirección que el pool ya satura.
    dif_fav < 0  ⇒ lo subvalúa (el caso clásico del favorito-longshot en casas
                   recreativas es este: la longshot está sobreprecida).

Acumula en data/odds_compare/ para que la conclusión no dependa de una fecha: con
8 partidos por fecha, hacen falta 3-4 fechas para separar sesgo de ruido.

## OJO — dónde corre

**Pinnacle bloquea las redes uruguayas**: desde la Mac esto NO anda. Corre en el
droplet de NYC:

    ssh root@159.203.66.24 'cd /opt/penca && .venv/bin/python -m scripts.odds_sharp_vs_supermatch'

La parte de Supermatch sí anda desde cualquier lado (`--solo-supermatch` reporta el
vig y el contraste proportional/Shin sin tocar Pinnacle).

## Qué NO hace

No cambia nada de producción. Es medición: si aparece un sesgo estable, recién ahí
se decide entre anclar λ a Pinnacle, meterlo como tercer componente del blend, o
solo corregir el de-vig — y esa decisión se mide con Δ E[premio] pareado, como todas.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.model.market_probs import devig, overround

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "odds_compare"

# Nombre de la liga uruguaya en Pinnacle. Se matchea por substring en minúsculas
# contra el listado de ligas de fútbol, porque el id cambia entre temporadas y
# descubrirlo por nombre es más robusto que hardcodearlo.
LIGA_UY_HINTS = ("uruguay",)
SPORT_SOCCER = 29                      # config/valuebet.yaml → sport_ids.soccer

# Distancia máxima entre los kickoffs de los dos books para aceptar un match.
MAX_DELTA_HORAS = 6.0


@dataclass
class ParMatcheado:
    partido: str
    start_utc: str
    vig_supermatch: float
    vig_pinnacle: float
    # probabilidades de-vigueadas, ordenadas por ROL según Pinnacle
    sm_fav: float
    pin_fav: float
    sm_empate: float
    pin_empate: float
    sm_dog: float
    pin_dog: float
    favorito: str                      # "home" | "away"

    @property
    def dif_fav(self) -> float:
        return self.sm_fav - self.pin_fav

    @property
    def dif_empate(self) -> float:
        return self.sm_empate - self.pin_empate

    @property
    def dif_dog(self) -> float:
        return self.sm_dog - self.pin_dog


# -------------------- matching --------------------

# Tokens que NO distinguen equipos: sufijos de club, ciudad y localizadores que un
# book pone y el otro no. Todo token fuera de esta lista SÍ distingue — ver `_similar`.
_RUIDO = {"fc", "cf", "ac", "sc", "cd", "club", "atletico", "atlético", "uru",
          "uruguay", "montevideo", "sporting", "de", "las", "piedras", "(uru)"}


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(c for c in s.replace("(", " ").replace(")", " ").split())


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if t not in _RUIDO}


def _similar(a: str, b: str) -> float:
    """Similitud por tokens, con los calificativos tratados como DISCRIMINANTES.

    El caso que este matcher existe para no repetir: `_norm("Cerro") in
    _norm("Cerro Largo")`. Cualquier score basado en substring o en subconjunto los
    da por iguales — y son dos clubes distintos. Acá, si a uno le sobra un token
    significativo (uno que no está en `_RUIDO`), NO son el mismo equipo: 0.0.

    "Liverpool (URU)" vs "Liverpool Montevideo" sí matchea, porque lo que sobra de
    los dos lados es ruido conocido.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    # tokens de cada lado sin contraparte (permitiendo abreviaturas: "penarol"/"penar")
    def _huerfanos(x: set[str], y: set[str]) -> set[str]:
        return {t for t in x
                if not any(t == u or (len(t) > 3 and t in u) or (len(u) > 3 and u in t)
                           for u in y)}

    if _huerfanos(ta, tb) or _huerfanos(tb, ta):
        return 0.0
    return 1.0


def emparejar(sm_eventos, pin_eventos, umbral: float = 0.5):
    """[(evento_supermatch, evento_pinnacle)] por nombre de equipos + hora cercana.

    Deliberadamente conservador: ante duda NO empareja. Un match equivocado mete un
    partido distinto en la comparación y produce un "sesgo" que es puro error de
    matcheo — el mismo modo de falla que el cruce Cerro/Cerro Largo de agosto.
    """
    out = []
    for sm in sm_eventos:
        t_sm = datetime.fromisoformat(sm["start_utc"].replace("Z", "+00:00"))
        mejor, mejor_score = None, 0.0
        for pin in pin_eventos:
            t_pin = datetime.fromisoformat(pin["start_utc"].replace("Z", "+00:00"))
            if abs((t_sm - t_pin).total_seconds()) > MAX_DELTA_HORAS * 3600:
                continue
            # MIN y no promedio: un lado que no matchea tiene que ser fatal. Con el
            # promedio, un visitante idéntico rescataba un local equivocado —
            # "Cerro vs Albion" contra "Cerro Largo vs Albion" daba 0.5 y entraba.
            s = min(_similar(sm["home"], pin["home"]), _similar(sm["away"], pin["away"]))
            if s > mejor_score:
                mejor, mejor_score = pin, s
        if mejor is not None and mejor_score >= umbral:
            out.append((sm, mejor))
        else:
            log.warning("sin match en Pinnacle: %s vs %s (mejor score %.2f)",
                        sm["home"], sm["away"], mejor_score)
    return out


def comparar_par(sm: dict, pin: dict, metodo_sm: str = "proportional") -> ParMatcheado:
    """Un partido matcheado → probabilidades de-vigueadas alineadas por ROL.

    Pinnacle siempre se de-viguea con Shin (es la referencia sharp y su vig es chico,
    así que el método casi no importa de su lado); Supermatch con el método que use
    producción, que es lo que se está auditando.
    """
    p_sm = devig(sm["x1x2"], metodo_sm)
    p_pin = devig(pin["x1x2"], "shin")
    favorito = "home" if p_pin["home"] >= p_pin["away"] else "away"
    dog = "away" if favorito == "home" else "home"
    return ParMatcheado(
        partido=f'{sm["home"]} vs {sm["away"]}',
        start_utc=sm["start_utc"],
        vig_supermatch=overround(sm["x1x2"]) - 1.0,
        vig_pinnacle=overround(pin["x1x2"]) - 1.0,
        sm_fav=p_sm[favorito], pin_fav=p_pin[favorito],
        sm_empate=p_sm["draw"], pin_empate=p_pin["draw"],
        sm_dog=p_sm[dog], pin_dog=p_pin[dog],
        favorito=favorito,
    )


# -------------------- fetch --------------------

def supermatch_eventos() -> list[dict]:
    """1X2 de la Primera uruguaya, del MISMO módulo que usa producción."""
    from src.clausura.odds import fetch_primera_odds

    out = []
    for e in fetch_primera_odds():
        if e.x1x2 and len(e.x1x2) == 3:
            out.append({"home": e.home, "away": e.away, "start_utc": e.start_utc,
                        "x1x2": dict(e.x1x2),
                        # el over 2.5 entra al fit de λ igual que en producción: sin
                        # él los λ de este reporte no son los que corren de verdad
                        "totals": {k: dict(v) for k, v in (e.totals or {}).items()}})
    return out


def pinnacle_eventos() -> list[dict]:
    """1X2 de la liga uruguaya en Pinnacle. Requiere red NO uruguaya."""
    from src.valuebet.books.pinnacle_vb import (
        american_to_decimal, get_markets, get_matchups, list_leagues,
    )

    ligas = [lg for lg in list_leagues(SPORT_SOCCER)
             if any(h in (lg.get("name", "") or "").lower() for h in LIGA_UY_HINTS)]
    if not ligas:
        log.error("Pinnacle no lista ninguna liga uruguaya de fútbol")
        return []
    log.info("ligas uruguayas en Pinnacle: %s",
             ", ".join(f'{lg["name"]} (id {lg["id"]})' for lg in ligas))

    info: dict[int, dict] = {}
    for lg in ligas:
        for m in get_matchups(lg["id"]):
            if m.get("type") != "matchup" or not m.get("startTime"):
                continue
            parts = m.get("participants", [])
            home = next((p["name"] for p in parts if p.get("alignment") == "home"), None)
            away = next((p["name"] for p in parts if p.get("alignment") == "away"), None)
            # sub-mercados con calificador entre paréntesis (corners, tarjetas) NO
            # son el resultado del partido
            if not (home and away) or "(" in home or "(" in away:
                continue
            info[m["id"]] = {"home": home, "away": away, "start_utc": m["startTime"]}

    out = []
    for lg in ligas:
        for mk in get_markets(lg["id"]):
            if mk.get("type") != "moneyline" or mk.get("period", 0) != 0:
                continue
            ev = info.get(mk.get("matchupId"))
            if ev is None:
                continue
            precios = {p["designation"]: american_to_decimal(p["price"])
                       for p in mk.get("prices", []) if p.get("price") is not None}
            if {"home", "draw", "away"} <= set(precios):
                out.append({**ev, "x1x2": {k: precios[k] for k in ("home", "draw", "away")}})
    return out


# -------------------- reporte --------------------

def resumir(pares: list[ParMatcheado]) -> str:
    if not pares:
        return "sin partidos matcheados."
    fav = np.array([p.dif_fav for p in pares])
    emp = np.array([p.dif_empate for p in pares])
    dog = np.array([p.dif_dog for p in pares])
    n = len(pares)

    lines = [f"{'partido':42} {'vigSM':>6} {'vigPin':>6} {'fav SM':>7} {'fav Pin':>7} {'dif':>7}"]
    for p in pares:
        lines.append(f"{p.partido[:41]:42} {p.vig_supermatch:6.1%} {p.vig_pinnacle:6.1%} "
                     f"{p.sm_fav:7.3f} {p.pin_fav:7.3f} {p.dif_fav:+7.3f}")

    def linea(nombre: str, v: np.ndarray) -> str:
        se = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        t = float(np.mean(v) / se) if se > 0 else 0.0
        return (f"  {nombre:<12} {np.mean(v):+.4f} ± {se:.4f}  (t={t:+.2f}, "
                f"{int((v > 0).sum())}/{len(v)} positivos)")

    lines += ["", f"AGREGADO sobre {n} partidos — Supermatch menos Pinnacle:",
              linea("favorito", fav), linea("empate", emp), linea("no-favorito", dog), ""]
    se_fav = float(np.std(fav, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if n >= 8 and se_fav > 0 and abs(float(np.mean(fav))) > 2 * se_fav:
        direccion = ("SOBREVALÚA al favorito ⇒ nuestras grillas cargan de más al chalk"
                     if float(np.mean(fav)) > 0 else
                     "SUBVALÚA al favorito ⇒ estamos jugando menos al chalk de lo debido")
        lines.append(f"⚠️ Sesgo estable: Supermatch {direccion}. "
                     f"Vale medir Δ E[premio] pareado anclando λ a Pinnacle.")
    else:
        lines.append(f"Sin sesgo distinguible del ruido todavía "
                     f"({n} partidos; hacen falta ~3-4 fechas para concluir).")
    lines.append(f"Vig medio: Supermatch {np.mean([p.vig_supermatch for p in pares]):.1%} · "
                 f"Pinnacle {np.mean([p.vig_pinnacle for p in pares]):.1%}")
    return "\n".join(lines)


def acumular(pares: list[ParMatcheado], persistir: bool = True) -> list[dict]:
    """Suma esta corrida al historial, deduplicando por (partido, start_utc)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "sharp_vs_supermatch.json"
    hist: dict[tuple[str, str], dict] = {}
    if path.exists():
        try:
            for d in json.loads(path.read_text(encoding="utf-8")).get("pares", []):
                hist[(d["partido"], d["start_utc"])] = d
        except Exception:                                      # noqa: BLE001
            hist = {}
    for p in pares:
        hist[(p.partido, p.start_utc)] = {**asdict(p), "medido_utc":
                                          datetime.now(timezone.utc).isoformat()}
    out = [hist[k] for k in sorted(hist)]
    if persistir:
        path.write_text(json.dumps({"pares": out}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return out


# -------------------- solo-Supermatch (corre desde cualquier lado) --------------------

def reporte_devig_supermatch() -> str:
    """Vig y contraste proportional/Shin sin tocar Pinnacle.

    Es la mitad de la auditoría que no necesita red no-uruguaya: cuantifica el
    overround y cuánto cambia λ según el método de de-vig.
    """
    from src.clausura.economics import MAX_GOALS, index_score
    from src.model.poisson import MarketConstraints, fit_params, score_grid

    def lams(ev, metodo):
        """Mismo camino que picks.market_lambdas: 1X2 + over 2.5 si está."""
        p = devig(ev["x1x2"], metodo)
        totals = ev.get("totals") or {}
        o25 = devig(totals["2.5"], metodo).get("over") if "2.5" in totals else None
        return fit_params(MarketConstraints(p_home_win=p["home"], p_draw=p["draw"],
                                            p_away_win=p["away"], p_over_2_5=o25))

    evs = supermatch_eventos()
    lines = [f"{'partido':40} {'vig':>6} {'difFav':>7} {'λL prop':>8} {'λL shin':>8} modal",
             ""]
    cambios = 0
    for ev in evs:
        pp, ps = devig(ev["x1x2"], "proportional"), devig(ev["x1x2"], "shin")
        fav = max(pp, key=pp.get)
        lp, ls = lams(ev, "proportional"), lams(ev, "shin")
        gp = score_grid(lp[0], lp[1], lp[2], max_goals=MAX_GOALS).ravel()
        gs = score_grid(ls[0], ls[1], ls[2], max_goals=MAX_GOALS).ravel()
        mp, ms = index_score(int(np.argmax(gp))), index_score(int(np.argmax(gs)))
        marca = ""
        if mp != ms:
            cambios += 1
            marca = f"  {mp[0]}-{mp[1]} → {ms[0]}-{ms[1]}  <<< CAMBIA"
        lines.append(f'{(ev["home"][:19] + " v " + ev["away"][:17]):40} '
                     f'{overround(ev["x1x2"]) - 1:6.1%} {pp[fav] - ps[fav]:+7.4f} '
                     f'{lp[0]:8.3f} {ls[0]:8.3f} {marca or f"  {mp[0]}-{mp[1]}"}')
    lines += ["", f"El de-vig cambia el marcador modal en {cambios}/{len(evs)} partidos.",
              "Producción usa proportional en el 1X2 (picks.market_lambdas)."]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-supermatch", action="store_true",
                    help="vig y de-vig sin Pinnacle (anda desde Uruguay)")
    ap.add_argument("--metodo", default="proportional", choices=("proportional", "shin"),
                    help="de-vig del lado Supermatch (default: el de producción)")
    ap.add_argument("--dry-run", action="store_true", help="no persistir el historial")
    args = ap.parse_args()

    if args.solo_supermatch:
        print(reporte_devig_supermatch())
        return

    sm = supermatch_eventos()
    log.info("Supermatch: %d partidos con 1X2", len(sm))
    pin = pinnacle_eventos()
    log.info("Pinnacle: %d partidos con 1X2", len(pin))
    if not pin:
        print("Sin datos de Pinnacle. ¿Estás corriendo desde una red uruguaya? "
              "Pinnacle las bloquea — usá el droplet de NYC, o --solo-supermatch.")
        return

    pares = [comparar_par(a, b, args.metodo) for a, b in emparejar(sm, pin)]
    print(resumir(pares))
    todos = acumular(pares, persistir=not args.dry_run)
    if not args.dry_run:
        print(f"\nhistorial acumulado: {len(todos)} partidos en "
              f"{OUT_DIR / 'sharp_vs_supermatch.json'}")


if __name__ == "__main__":
    main()

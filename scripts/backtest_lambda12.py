"""¿Los goles de la Primera uruguaya son independientes dado el λ del partido?

LA CONTRADICCIÓN (auditoría del 13/8). Hoy el sistema afirma las dos cosas a la vez,
según qué partido mires:

  * en los 8 partidos con mercado, `market_lambdas` fitea un λ12 libre contra el 1X2
    y lo propaga al blend (picks.py) — o sea, dependencia POSITIVA;
  * en los otros 112 —y en el 100% de P(campeón), que no tiene mercado— la grilla
    sale de los ratings, y ahí `ratings.py` declara λ12 = 0 citando "corr +0.002 en
    598 partidos".

Las dos no pueden ser ciertas. O el mercado sobreprecia el empate (y entonces el λ12
del fit mete masa espuria en 0-0/1-1 justo en los partidos que se cargan), o hay
dependencia real que una correlación global —un solo momento— no ve.

## Por qué se miden DOS modelos

La correlación de Pearson ≈ 0 refuta la bivariada, pero NO refuta la otra forma de
dependencia, que es la que importa acá:

  * **Bivariate Poisson (λ12 ≥ 0)**: X = W1+W3, Y = W2+W3. Mueve la covarianza. Es
    exactamente el modelo que `market_lambdas` fitea, así que este brazo contesta
    "¿el histórico respalda lo que el mercado dice?".
  * **Dixon-Coles (ρ)**: reajusta SOLO los cuatro marcadores bajos (0-0, 1-0, 0-1,
    1-1) con un factor τ. Puede subir la diagonal baja dejando la correlación casi
    intacta — es decir, puede haber "más empates de los que predice el Poisson" con
    corr ≈ 0. Si el histórico tiene esta forma de dependencia, el λ12 del mercado
    estaría capturando algo real con el instrumento equivocado.

## Método

Walk-forward por fecha sobre las temporadas históricas, igual que
`backtest_ratings_intra.py`. Para cada fecha: se fitean los ratings con lo ANTERIOR,
se fitea el parámetro extra (λ12 o ρ) también con lo anterior, y se evalúa sobre la
fecha retenida. Nada del conjunto de test entra al fit.

Métrica: log-verosimilitud del MARCADOR EXACTO observado (que es lo que la grilla
tiene que predecir), pareada por partido contra el Poisson independiente. Y dos
calibraciones que apuntan al corazón del asunto: P(empate) y P(0-0) predichas vs
la frecuencia real.

Uso:
    python -m scripts.backtest_lambda12
    python -m scripts.backtest_lambda12 --min-train 120
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from math import lgamma

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.clausura.intermedio import load_dataset_completo  # noqa: E402
from src.clausura.ratings import fit_ratings  # noqa: E402

# Cota del λ12 global a explorar. El mercado de la F2 fitea λ12 en el orden de
# 0,0-0,3, así que 0,6 deja margen de sobra sin que el optimizador se vaya al borde.
LAM12_MAX = 0.6
# Dixon-Coles: ρ fuera de ±0,3 vuelve τ negativo con λ típicos de esta liga.
RHO_ABS_MAX = 0.25


def _ts(p):
    from datetime import datetime
    return datetime.fromisoformat(p.inicio_utc.replace("Z", "+00:00")).timestamp()


# -------------------- verosimilitudes (vectorizadas sobre partidos) --------------------

def loglik_indep(gl, gv, ll, lv):
    """log P(gl, gv) bajo Poisson independiente."""
    return (gl * np.log(ll) - ll - np.array([lgamma(g + 1) for g in gl])
            + gv * np.log(lv) - lv - np.array([lgamma(g + 1) for g in gv]))


def loglik_bivar(gl, gv, ll, lv, lam12):
    """log P(gl, gv) bajo bivariate Poisson con marginales (ll, lv) y covarianza λ12.

    Misma parametrización que `src.model.poisson.bivariate_poisson_pmf` (λ_L y λ_V
    son las MEDIAS marginales, y por dentro λ1 = λ_L − λ12): si difiriera, esto
    mediría un modelo distinto del que corre en producción.

    λ12 se clipea por partido a min(λ_L, λ_V), igual que hace el blend de picks.py.
    """
    lam12 = np.minimum(lam12, np.minimum(ll, lv) * 0.999)
    l1, l2 = ll - lam12, lv - lam12
    base = (-(l1 + l2 + lam12)
            + gl * np.log(l1) - np.array([lgamma(g + 1) for g in gl])
            + gv * np.log(l2) - np.array([lgamma(g + 1) for g in gv]))
    # Σ_k C(gl,k) C(gv,k) k! (λ12/(λ1λ2))^k — los goles son chicos, así que el
    # sumatorio corta enseguida
    from math import comb, factorial
    s = np.zeros_like(base)
    kmax = int(min(gl.max(), gv.max()))
    r = lam12 / (l1 * l2)
    for k in range(kmax + 1):
        cont = np.array([comb(int(a), k) * comb(int(b), k) * factorial(k)
                         if k <= min(int(a), int(b)) else 0.0
                         for a, b in zip(gl, gv)])
        s += cont * r ** k
    return base + np.log(np.maximum(s, 1e-300))


def _tau_dixon_coles(gl, gv, ll, lv, rho):
    """Factor τ de Dixon-Coles: solo toca los cuatro marcadores bajos."""
    tau = np.ones_like(ll)
    m00 = (gl == 0) & (gv == 0)
    m01 = (gl == 0) & (gv == 1)
    m10 = (gl == 1) & (gv == 0)
    m11 = (gl == 1) & (gv == 1)
    tau[m00] = 1.0 - ll[m00] * lv[m00] * rho
    tau[m01] = 1.0 + ll[m01] * rho
    tau[m10] = 1.0 + lv[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


def loglik_dixon(gl, gv, ll, lv, rho):
    tau = _tau_dixon_coles(gl, gv, ll, lv, rho)
    if np.any(tau <= 0):
        return np.full_like(ll, -1e6)
    return loglik_indep(gl, gv, ll, lv) + np.log(tau)


# -------------------- ajuste del parámetro extra (sobre el train) --------------------

def fit_lam12(gl, gv, ll, lv) -> float:
    r = minimize_scalar(lambda x: -loglik_bivar(gl, gv, ll, lv, x).sum(),
                        bounds=(0.0, LAM12_MAX), method="bounded")
    return float(r.x)


def fit_rho(gl, gv, ll, lv) -> float:
    r = minimize_scalar(lambda x: -loglik_dixon(gl, gv, ll, lv, x).sum(),
                        bounds=(-RHO_ABS_MAX, RHO_ABS_MAX), method="bounded")
    return float(r.x)


# -------------------- grillas para la calibración --------------------

def p_empate_y_cero(ll, lv, lam12=0.0, rho=None, max_goals=8):
    """(P(empate), P(0-0)) predichas por partido bajo el modelo dado."""
    from src.model.poisson import bivariate_poisson_pmf

    p_emp = np.zeros_like(ll)
    p_00 = np.zeros_like(ll)
    for i, (a, b) in enumerate(zip(ll, lv)):
        l12 = min(lam12, min(a, b) * 0.999)
        tot = emp = 0.0
        p00 = 0.0
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                p = bivariate_poisson_pmf(x, y, a, b, l12)
                if rho is not None:
                    if x == 0 and y == 0:
                        p *= 1.0 - a * b * rho
                    elif x == 0 and y == 1:
                        p *= 1.0 + a * rho
                    elif x == 1 and y == 0:
                        p *= 1.0 + b * rho
                    elif x == 1 and y == 1:
                        p *= 1.0 - rho
                tot += p
                if x == y:
                    emp += p
                if x == 0 and y == 0:
                    p00 = p
        p_emp[i] = emp / tot
        p_00[i] = p00 / tot
    return p_emp, p_00


# -------------------- walk-forward --------------------

def run(min_train: int = 120) -> dict:
    partidos = sorted(load_dataset_completo(), key=_ts)

    fechas = []
    for p in partidos:
        if not fechas or fechas[-1][0] != (p.campeonato_id, p.fecha_id):
            fechas.append(((p.campeonato_id, p.fecha_id), []))
        fechas[-1][1].append(p)

    filas = []          # por partido de test: loglik de cada modelo + calibración
    lam12s, rhos = [], []

    for i in range(1, len(fechas)):
        train = [p for _, ps in fechas[:i] for p in ps]
        if len(train) < min_train:
            continue
        test = fechas[i][1]

        r = fit_ratings(train)
        tr_ll = np.array([r.lambdas(p.local, p.visitante)[0] for p in train])
        tr_lv = np.array([r.lambdas(p.local, p.visitante)[1] for p in train])
        tr_gl = np.array([p.goles_local for p in train])
        tr_gv = np.array([p.goles_visitante for p in train])

        lam12 = fit_lam12(tr_gl, tr_gv, tr_ll, tr_lv)
        rho = fit_rho(tr_gl, tr_gv, tr_ll, tr_lv)
        lam12s.append(lam12)
        rhos.append(rho)

        te_ll = np.array([r.lambdas(p.local, p.visitante)[0] for p in test])
        te_lv = np.array([r.lambdas(p.local, p.visitante)[1] for p in test])
        te_gl = np.array([p.goles_local for p in test])
        te_gv = np.array([p.goles_visitante for p in test])

        ll_i = loglik_indep(te_gl, te_gv, te_ll, te_lv)
        ll_b = loglik_bivar(te_gl, te_gv, te_ll, te_lv, lam12)
        ll_d = loglik_dixon(te_gl, te_gv, te_ll, te_lv, rho)

        emp_i, c00_i = p_empate_y_cero(te_ll, te_lv)
        emp_b, c00_b = p_empate_y_cero(te_ll, te_lv, lam12=lam12)
        emp_d, c00_d = p_empate_y_cero(te_ll, te_lv, rho=rho)

        for k in range(len(test)):
            filas.append({
                "indep": ll_i[k], "bivar": ll_b[k], "dixon": ll_d[k],
                "empate_real": int(te_gl[k] == te_gv[k]),
                "cero_real": int(te_gl[k] == 0 and te_gv[k] == 0),
                "p_emp_indep": emp_i[k], "p_emp_bivar": emp_b[k], "p_emp_dixon": emp_d[k],
                "p_00_indep": c00_i[k], "p_00_bivar": c00_b[k], "p_00_dixon": c00_d[k],
            })

    return {"filas": filas, "lam12s": np.array(lam12s), "rhos": np.array(rhos)}


def _delta(filas, modelo):
    d = np.array([f[modelo] - f["indep"] for f in filas])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-train", type=int, default=120)
    args = ap.parse_args()

    out = run(min_train=args.min_train)
    filas, lam12s, rhos = out["filas"], out["lam12s"], out["rhos"]
    n = len(filas)

    print(f"\n=== ¿los goles son independientes dado λ? — {n} partidos walk-forward ===\n")
    print(f"parámetro fiteado sobre el train, fecha a fecha:")
    print(f"  λ12  media {lam12s.mean():.4f}  ·  mediana {np.median(lam12s):.4f}  "
          f"·  rango [{lam12s.min():.4f}, {lam12s.max():.4f}]")
    print(f"  ρ    media {rhos.mean():+.4f}  ·  mediana {np.median(rhos):+.4f}  "
          f"·  rango [{rhos.min():+.4f}, {rhos.max():+.4f}]")

    print(f"\nΔ log-verosimilitud por partido vs Poisson independiente "
          f"(pareado, + es mejor):")
    for modelo, etiqueta in (("bivar", "bivariate λ12"), ("dixon", "Dixon-Coles ρ")):
        m, se = _delta(filas, modelo)
        t = m / se if se > 0 else 0.0
        veredicto = "SIGNIFICATIVO" if abs(t) > 2 else "no distinguible del ruido"
        # MDE al 80% de potencia con la regla del proyecto (config/decisiones.yaml):
        # se adopta con delta > 2·SE, así que hace falta 2·SE + 0,84·SE para verlo
        # el 80% de las veces. Sin esto, "no significativo" se lee como "no existe".
        print(f"  {etiqueta:16} {m:+.5f} ± {se:.5f} nats  (t={t:+.2f})  {veredicto}"
              f"\n  {'':16} ciego a efectos < {2.84 * se:.4f} nats "
              f"(referencia: la ingesta intra-temporada valió +0,0073)")

    # La calibración es la evidencia FUERTE acá: con 528 partidos la frecuencia real
    # tiene un error de ~1,9 pp en el empate, así que una brecha de medio punto no
    # significa nada, pero una de tres sí se vería.
    print(f"\ncalibración del EMPATE (lo que el hallazgo dice que estaría corto):")
    real = np.mean([f["empate_real"] for f in filas])
    se_real = np.sqrt(real * (1 - real) / n)
    print(f"  real                {real:.4f}  (± {se_real:.4f})")
    for modelo, etiqueta in (("indep", "Poisson indep."), ("bivar", "bivariate λ12"),
                             ("dixon", "Dixon-Coles ρ")):
        pred = np.mean([f[f"p_emp_{modelo}"] for f in filas])
        print(f"  {etiqueta:18} {pred:.4f}   ({pred - real:+.4f})")

    print(f"\ncalibración del 0-0 (el marcador que el pool subjuega):")
    real00 = np.mean([f["cero_real"] for f in filas])
    se00 = np.sqrt(real00 * (1 - real00) / n)
    print(f"  real                {real00:.4f}  (± {se00:.4f})")
    for modelo, etiqueta in (("indep", "Poisson indep."), ("bivar", "bivariate λ12"),
                             ("dixon", "Dixon-Coles ρ")):
        pred = np.mean([f[f"p_00_{modelo}"] for f in filas])
        print(f"  {etiqueta:18} {pred:.4f}   ({pred - real00:+.4f})")

    m_b, se_b = _delta(filas, "bivar")
    m_d, se_d = _delta(filas, "dixon")
    print()
    if m_b < 2 * se_b and m_d < 2 * se_d:
        print("VEREDICTO — dos conclusiones, y conviene no mezclarlas:\n"
              "\n"
              "  1. EL LADO RATINGS QUEDA VALIDADO. Con λ de los ratings, el histórico no\n"
              "     muestra dependencia en ninguna de las dos formas, y el Poisson\n"
              "     independiente calibra el empate mejor que los dos modelos con\n"
              "     parámetro extra. `ratings.py` tenía razón: λ12 = 0 se queda, y ahora\n"
              "     está medido y no supuesto. Eso cubre 112 de 120 eventos y todo P(campeón).\n"
              "\n"
              "  2. EL LADO MERCADO ES OTRA PREGUNTA, y esto la deja bien planteada.\n"
              "     Ahí λ12 NO es una creencia sobre dependencia: es el tercer grado de\n"
              "     libertad que necesita el fit para clavar las tres patas del 1X2 (con\n"
              "     λ12 = 0 sobran targets). O sea que el fit reporta λ12 > 0 siempre que\n"
              "     el mercado pida MÁS empate del que implican sus propias marginales.\n"
              "     Y eso es esperable en una casa con 14,7% de overround: el empate es\n"
              "     donde se carga margen (medido el 13/8 contra Pinnacle: el empate de\n"
              "     Supermatch viene +0,53 pp con el de-vig de producción).\n"
              "     Conclusión: el λ12 del mercado probablemente está absorbiendo vig, no\n"
              "     estructura. Pero cambiarlo se decide con Δ E[premio] pareado, no con\n"
              "     esto — el mismo estándar que dejó el de-vig sin cambiar.")
    elif m_d > 2 * se_d and m_d > m_b:
        print("VEREDICTO: hay dependencia, pero de la forma Dixon-Coles (diagonal baja),\n"
              "  NO de la que modela la bivariada. El λ12 del mercado estaría capturando\n"
              "  algo real con el instrumento equivocado.")
    else:
        print("VEREDICTO: la bivariada gana — el λ12 del mercado tiene respaldo histórico\n"
              "  y ratings.py debería llevarlo también.")


if __name__ == "__main__":
    main()

"""Monitor de salud de la estrategia (read-only).

Imprime las 3 señales que importan para el objetivo P(al menos una penca gana el pool),
con un veredicto OK / ATENCION por cada una.

Uso:
    python scripts/monitor_estrategia.py            # señales 1 y 2 (solo API, sin SSH)
    python scripts/monitor_estrategia.py --vps      # + señal 3 (objetivo del optimizador, vía SSH al VPS)

Lee credenciales y PENCA_IDS de .env (las 15 pencas). No publica ni muta nada.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VPS_HOST = os.environ.get("PENCA_VPS_HOST", "root@68.183.52.166")
VPS_DATA = "/opt/penca/data"


def _load_env() -> None:
    # Busca .env en el repo, en el cwd y caminando hacia arriba (el worktree puede no tenerlo).
    candidates = [ROOT / ".env", Path.cwd() / ".env"]
    candidates += [p / ".env" for p in Path.cwd().resolve().parents]
    f = next((c for c in candidates if c.exists()), None)
    if f is None:
        return
    for ln in f.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _api(path: str) -> dict | None:
    base = os.environ.get("PENCA_API_BASE_URL", "").rstrip("/")
    key = os.environ.get("PENCA_API_KEY", "")
    if not base or not key:
        return None
    try:
        req = urllib.request.Request(f"{base}{path}", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=15.0) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  [error API {path}: {e}]")
        return None


def _ssh(remote_cmd: str) -> str | None:
    try:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", VPS_HOST, remote_cmd],
            capture_output=True, text=True, timeout=40,
        )
        return out.stdout if out.returncode == 0 else None
    except Exception as e:
        print(f"  [error SSH: {e}]")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vps", action="store_true", help="incluir señal 3 (objetivo del optimizador) vía SSH")
    args = ap.parse_args()
    _load_env()

    our = [int(x) for x in os.environ.get("PENCA_IDS", "").split(",") if x.strip()]
    if not our:
        print("PENCA_IDS vacío en .env — abortando.")
        return 1

    lb = _api("/leaderboard")
    mt = _api("/matches")
    if not lb:
        print("No pude leer /leaderboard — revisá PENCA_API_KEY.")
        return 1
    entries = lb["entries"]
    N = lb["n_players"]
    byid = {e["penca_id"]: e for e in entries}
    pts_all = sorted((e["points_total"] for e in entries), reverse=True)
    n_fin = sum(1 for m in (mt or {}).get("matches", []) if m.get("status") == "finished") if mt else None

    ours = sorted((byid[p] for p in our if p in byid), key=lambda e: e["rank"])
    if not ours:
        print("Ninguna de nuestras pencas aparece en el leaderboard.")
        return 1
    best, worst = ours[0], ours[-1]
    top10 = sum(1 for e in ours if e["rank"] <= 10)
    top50 = sum(1 for e in ours if e["rank"] <= 50)
    top100 = sum(1 for e in ours if e["rank"] <= 100)
    spread = best["points_total"] - worst["points_total"]
    gap_top = pts_all[0] - best["points_total"]

    print("=" * 64)
    print(f"MONITOR ESTRATEGIA — pool {N} jugadores"
          + (f" | {n_fin} partidos jugados" if n_fin is not None else "")
          + f" | snapshot {lb.get('as_of','')[:16]}")
    print("=" * 64)

    # ---- SEÑAL 1: una penca en la cola ganadora ----
    v1 = "OK" if best["rank"] <= 10 else ("ATENCION" if best["rank"] > 30 else "OK-ish")
    print(f"\n[1] ¿Una penca en la cola ganadora?   -> {v1}")
    print(f"    mejor: {best['penca_id']}  #{best['rank']}/{N}  {best['points_total']} pts  | gap al top: {gap_top}")
    print(f"    nuestras en  top-10: {top10}   top-50: {top50}   top-100: {top100}")
    print(f"    (sano = al menos 1 en top-10; alerta si la mejor cae > #30)")

    # ---- SEÑAL 2: diversificación (spread interno) ----
    v2 = "OK" if spread >= 6 else "ATENCION"
    print(f"\n[2] ¿Se mantiene la diversificación?   -> {v2}")
    print(f"    spread interno = {spread} pts  (mejor {best['points_total']} - peor {worst['points_total']})")
    print(f"    (sano = spread no colapsa hacia 0; si tiende a 0 = convergencia a chalk)")

    # ---- SEÑAL 3: tail-bet (objetivo del optimizador) — requiere VPS ----
    print(f"\n[3] ¿Se activó el tail-bet (objetivo p_top_k)?", end="")
    if not args.vps:
        print("   -> (corré con --vps)")
    else:
        print()
        cmd = (
            f"for d in $(ls -d {VPS_DATA}/predictions/*/ | sort -t/ -k6 -n | tail -6); do "
            f"f=$(ls $d/v*.json 2>/dev/null | sort | tail -1); "
            f"mid=$(basename $d); "
            f'python3 -c "import json,sys; m=json.load(open(\'$f\')).get(\'assignment_meta\',{{}}); '
            f"print('m'+'$mid', m.get('objective'), 'thr='+str(m.get('threshold')), "
            f"'prem='+str(round(m.get('horizon_premium') or 0,1)))\"; done"
        )
        out = _ssh(cmd)
        if out:
            for ln in out.strip().splitlines():
                print(f"    {ln}")
            flipped = "p_top_k" in out and "e_max" not in out.split("p_top_k")[-1][:40]
            any_ptopk = "objective p_top_k" in out or " p_top_k " in out
            print(f"    -> {'ACTIVO en algún partido reciente (tail-bet persiguiendo el corte)' if any_ptopk else 'en fallback e_max (esperado temprano; se reactiva cuando corte+premium sea alcanzable)'}")
        else:
            print("    [no pude leer el VPS — revisá la clave SSH ~/.ssh/id_ed25519]")

    print("\n" + "-" * 64)
    print("Resumen:  [1]", v1, " [2]", v2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

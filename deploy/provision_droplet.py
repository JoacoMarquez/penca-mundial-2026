#!/usr/bin/env python3
"""Provisiona el droplet de DigitalOcean para correr la pipeline de la penca.

Pasos automatizados:
    1. Verifica que el DO API token esté en .env.
    2. Sube la SSH key pública del usuario a DigitalOcean (si no está ya).
    3. Crea un droplet Basic Ubuntu 24.04 con la SSH key asociada.
    4. Espera a que esté "active" y devuelve la IP pública.

Después del provisioning, el flujo es manual (1 sola vez):
    ssh root@<ip> "curl -fsSL https://raw.githubusercontent.com/<repo>/main/deploy/setup_droplet.sh | bash"

Uso:
    python deploy/provision_droplet.py [--region nyc3] [--size s-1vcpu-1gb] [--name penca-mundial-2026]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx


DO_API = "https://api.digitalocean.com/v2"


def load_token() -> str:
    """Lee DO_API_TOKEN del .env (sin usar python-dotenv para evitar dependencia)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        print(f"FATAL: no existe {env_path}", file=sys.stderr)
        sys.exit(1)
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("DO_API_TOKEN="):
            token = line.split("=", 1)[1].strip()
            if not token or token.startswith("dop_v1_xxx"):
                print("FATAL: DO_API_TOKEN no está completado en .env", file=sys.stderr)
                sys.exit(1)
            return token
    print("FATAL: DO_API_TOKEN no encontrado en .env", file=sys.stderr)
    sys.exit(1)


def find_ssh_pubkey() -> tuple[str, str]:
    """Busca una SSH key pública en ~/.ssh/. Prioridad: ed25519 > rsa.
    Returns (label, key_content). Si no existe ninguna, instruye al usuario y aborta.
    """
    home = Path.home() / ".ssh"
    for fname in ("id_ed25519.pub", "id_rsa.pub"):
        p = home / fname
        if p.exists():
            return (fname, p.read_text().strip())
    print(
        "FATAL: no se encontró SSH key en ~/.ssh/. Generá una con:\n"
        "  ssh-keygen -t ed25519 -C 'penca-mundial' -f ~/.ssh/id_ed25519\n"
        "  (apretá Enter dos veces para no usar passphrase, o usá una)",
        file=sys.stderr,
    )
    sys.exit(1)


def ensure_ssh_key(client: httpx.Client, label: str, pubkey: str) -> int:
    """Sube la SSH key a DO si no está ya. Retorna su fingerprint id."""
    # Listar keys existentes
    resp = client.get(f"{DO_API}/account/keys")
    resp.raise_for_status()
    existing = resp.json().get("ssh_keys", [])
    for k in existing:
        if k["public_key"].strip() == pubkey:
            print(f"  ✓ SSH key ya estaba registrada: id={k['id']} fingerprint={k['fingerprint']}")
            return k["id"]

    # Crear
    resp = client.post(
        f"{DO_API}/account/keys",
        json={"name": f"penca-mundial-{label}", "public_key": pubkey},
    )
    resp.raise_for_status()
    key = resp.json()["ssh_key"]
    print(f"  ✓ SSH key subida: id={key['id']}")
    return key["id"]


def create_droplet(
    client: httpx.Client,
    name: str,
    region: str,
    size: str,
    ssh_key_id: int,
) -> int:
    """Crea el droplet. Retorna su id."""
    resp = client.post(
        f"{DO_API}/droplets",
        json={
            "name": name,
            "region": region,
            "size": size,
            "image": "ubuntu-24-04-x64",
            "ssh_keys": [ssh_key_id],
            "backups": False,
            "ipv6": False,
            "monitoring": True,
            "tags": ["penca-mundial-2026"],
        },
    )
    if resp.status_code >= 300:
        print(f"FATAL: error creando droplet:\n{resp.text}", file=sys.stderr)
        sys.exit(1)
    droplet = resp.json()["droplet"]
    print(f"  ✓ Droplet creado: id={droplet['id']} name={droplet['name']}")
    return droplet["id"]


def wait_for_active(client: httpx.Client, droplet_id: int, timeout: int = 180) -> dict:
    """Polling hasta que el droplet esté activo. Retorna el dict del droplet con su ip."""
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get(f"{DO_API}/droplets/{droplet_id}")
        resp.raise_for_status()
        d = resp.json()["droplet"]
        if d["status"] == "active":
            return d
        print(f"  ... status={d['status']}, esperando ({int(time.time() - start)}s)")
        time.sleep(5)
    print("FATAL: timeout esperando el droplet active", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="nyc3", help="DO region (default nyc3)")
    ap.add_argument("--size", default="s-1vcpu-1gb", help="DO size slug (default Basic $6/mes)")
    ap.add_argument("--name", default="penca-mundial-2026", help="Droplet name")
    args = ap.parse_args()

    token = load_token()
    label, pubkey = find_ssh_pubkey()

    with httpx.Client(
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=30.0,
    ) as client:
        print(f"› Subiendo SSH key ({label})…")
        key_id = ensure_ssh_key(client, label, pubkey)

        print(f"› Creando droplet ({args.size} en {args.region})…")
        droplet_id = create_droplet(client, args.name, args.region, args.size, key_id)

        print("› Esperando que el droplet esté active…")
        d = wait_for_active(client, droplet_id)

    ip = next(
        (n["ip_address"] for n in d["networks"]["v4"] if n["type"] == "public"),
        None,
    )

    print()
    print("=" * 60)
    print(f"✅ DROPLET PROVISIONADO")
    print(f"   Name : {d['name']}")
    print(f"   ID   : {d['id']}")
    print(f"   IP   : {ip}")
    print(f"   Size : {d['size_slug']}  ({args.region})")
    print(f"   Costo: ~$6/mes prorrateado por hora")
    print()
    print("Siguiente paso (SSH):")
    print(f"   ssh root@{ip}")
    print()
    print("Y dentro del droplet, correr:")
    print(f"   bash <(curl -fsSL <URL_GITHUB_RAW>/deploy/setup_droplet.sh)")
    print()
    print("Para destruir el droplet al final del Mundial:")
    print(f"   curl -X DELETE -H 'Authorization: Bearer $DO_API_TOKEN' \\\n"
          f"       {DO_API}/droplets/{d['id']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

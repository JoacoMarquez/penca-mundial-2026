# Deploy en DigitalOcean

## Provisioning (1 vez, desde tu Mac)

```bash
# 1. Asegurate de tener DO_API_TOKEN en .env (ya lo tenés)
# 2. Asegurate de tener una SSH key:
ssh-keygen -t ed25519 -C "penca-mundial" -f ~/.ssh/id_ed25519   # si no existe

# 3. Correr el provisioner (necesita httpx instalado localmente o usá uv/pipx)
python deploy/provision_droplet.py
```

El script:
1. Sube tu SSH pub key a DO (si no estaba).
2. Crea el droplet Basic ($6/mes, NYC3).
3. Espera a que esté `active` y te imprime la IP.

## Setup dentro del droplet (1 vez)

```bash
ssh root@<IP_DEL_DROPLET>

# Bajar el setup script (idempotente) y correrlo:
curl -fsSL https://raw.githubusercontent.com/<TU_USER>/<TU_REPO>/main/deploy/setup_droplet.sh -o setup.sh
chmod +x setup.sh
./setup.sh
```

Esto va a:
- Instalar Python 3.12, virtualenv, deps del sistema para Playwright.
- Clonar el repo en `/opt/penca`.
- Crear venv + instalar requirements.txt.
- Crear `/etc/penca/env` desde `.env.example` para que vos editás con tus secrets.
- Pausa ahí y te pide editar `/etc/penca/env`.

Cuando edites el env:
```bash
nano /etc/penca/env    # pegá las claves reales
./setup.sh             # correr de nuevo para instalar systemd units
```

## Operación

```bash
systemctl status penca-scheduler.timer       # estado del scheduler
systemctl status penca-scheduler.service     # última ejecución
journalctl -u penca-scheduler -f             # logs en vivo
tail -f /var/lib/penca/logs/scheduler.log    # logs del scheduler
```

Actualizar código:
```bash
cd /opt/penca && git pull
systemctl restart penca-scheduler.timer
```

## Destruir al final del Mundial

Desde tu Mac (ahorra $$ del prorrateo):
```bash
curl -X DELETE \
  -H "Authorization: Bearer $(grep DO_API_TOKEN ../.env | cut -d= -f2)" \
  https://api.digitalocean.com/v2/droplets/<DROPLET_ID>
```

## Costos esperados

- Droplet Basic ($6/mes): ~$8 USD para los 39 días del Mundial.
- Bandwidth: < 5 GB/mes, gratis.
- API calls a Anthropic: ~$5-15 totales (depende de cuántos partidos llegan a eliminatorias).
- API calls a Telegram: gratis.
- DO API: gratis.

**Total estimado:** $15-25 USD para todo el Mundial.

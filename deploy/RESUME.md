# Sistema apagado — cómo retomar

**Estado: APAGADO desde 2026-07-29.** El droplet `penca-mundial-2026` (id 573530640,
68.183.52.166) fue destruido. Facturación de DigitalOcean en **$0**.

Contexto: Mundial 2026 terminado (penca 1891 campeona, 1º/436). Valuebet quedó en
modo paper, nunca pasó a real.

---

## Qué se hizo (2026-07-29)

1. `deploy/shutdown_all.sh` en el VPS — bajó las 8 units (3 timers de penca,
   4 de valuebet, `penca-dashboard.service`). Verificado: 0 timers, 0 procesos.
2. `deploy/backup_vps.sh` — bajó 441 MB al Mac. Verificado 1552/1552 archivos.
3. `deploy/destroy_droplet.sh` — droplet destruido (HTTP 204, cuenta sin droplets).

## Dónde está el backup

```
/Users/joaquinmarquez/Documents/Personal/Automatizaciones/penca-backups/vps-20260729/
├── data/              # 430M — predicciones versionadas, ledger de valuebet,
│                      #        aliases, CLV, pool_snapshots, postmortems, dossiers
├── var-lib-penca/     # 11M — logs estructurados
├── etc-penca/
│   ├── env            # ⚠️ SECRETS: Anthropic, Telegram, penca API, odds
│   └── valuebet-env   # overrides no-secretos
├── penca-shutdown-state.txt   # qué units estaban habilitadas
├── timers-snapshot.txt
└── units-snapshot.txt
```

Está **fuera del repo** a propósito: contiene `/etc/penca/env` con las claves reales.
No lo muevas adentro del árbol de git.

---

## Retomar

El droplet ya no existe, así que hay que levantar uno nuevo. ~15 minutos.

```bash
# 1. Crear droplet (necesita DO_API_TOKEN en .env, ya está)
python deploy/provision_droplet.py
# → imprime la IP nueva. Guardala: export IP=<IP_NUEVA>
```

```bash
# 2. Setup base en el droplet nuevo
ssh root@$IP
curl -fsSL https://raw.githubusercontent.com/JoacoMarquez/penca-mundial-2026/main/deploy/setup_droplet.sh -o setup.sh
bash setup.sh    # clona /opt/penca, crea venv, deja /etc/penca/env vacío y corta ahí
```

```bash
# 3. Desde el Mac: restaurar secrets y datos
BK=/Users/joaquinmarquez/Documents/Personal/Automatizaciones/penca-backups/vps-20260729
scp   $BK/etc-penca/env           root@$IP:/etc/penca/env
scp   $BK/etc-penca/valuebet-env  root@$IP:/etc/penca/valuebet-env
rsync -az $BK/data/          root@$IP:/opt/penca/data/
rsync -az $BK/var-lib-penca/ root@$IP:/var/lib/penca/
```

```bash
# 4. En el droplet: instalar y habilitar units
ssh root@$IP
chmod 600 /etc/penca/env /etc/penca/valuebet-env
bash /opt/penca/deploy/setup_droplet.sh   # 2da pasada: instala units de penca
bash /opt/penca/deploy/setup_valuebet.sh  # instala y habilita valuebet
systemctl list-timers 'penca-*' 'valuebet-*'
```

```bash
# 5. Actualizar la IP hardcodeada en los helpers locales
#    deploy/health_check.sh:15  y  scripts/monitor_estrategia.py:18
#    (o exportar PENCA_VPS_HOST=root@$IP y no tocar nada)
```

**Si solo querés valuebet** (lo más probable — la penca ya no tiene torneo): saltate
`setup_droplet.sh` en el paso 4 y corré solo `setup_valuebet.sh`. Igual necesitás
restaurar `/etc/penca/env`, porque valuebet reusa esos secrets.

### Si en el futuro apagás con el droplet vivo

`shutdown_all.sh` guarda el estado en `/root/penca-shutdown-state.txt`, así que
revertir es un comando:

```bash
ssh root@<IP> 'bash /root/shutdown_all.sh --undo'
```

Re-habilita exactamente las units que estaban prendidas. Verificá que llegue el
heartbeat de Telegram.

---

## Deuda pendiente al momento de apagar

- La rama `quota-capture-analysis` tiene el hardening de valuebet (CLV fuzzy, flock,
  scan horario, matcher blindado) **sin PR mergeado y sin aplicar en el VPS**.
  Si retomás valuebet, mergear eso **antes** de re-desplegar.
- Valuebet nunca salió de paper — el ledger del backup es simulado, no plata real.

## Costos ahora

- Droplet: **$0** (destruido).
- Sin snapshot, así que tampoco hay costo de almacenamiento en DO.
- Anthropic / Telegram / Odds API: $0 mientras no corra nada.

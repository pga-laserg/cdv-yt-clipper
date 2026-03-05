#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/proxmox-storage-report.sh
# Optional env:
#   PROXMOX_HOST=proxmox
#   FULLRES_CT_ID=118
#   TUNNEL_CT_ID=102

PROXMOX_HOST="${PROXMOX_HOST:-proxmox}"
FULLRES_CT_ID="${FULLRES_CT_ID:-118}"
TUNNEL_CT_ID="${TUNNEL_CT_ID:-102}"

echo "== Proxmox Storage Report =="
echo "host=${PROXMOX_HOST} fullres_ct=${FULLRES_CT_ID} tunnel_ct=${TUNNEL_CT_ID}"
echo

echo "-- Proxmox node storage pools --"
ssh "${PROXMOX_HOST}" "pvesm status"
echo

echo "-- Proxmox ZFS summary --"
ssh "${PROXMOX_HOST}" "zpool list || true"
echo

echo "-- Full-res CT disk usage --"
ssh "${PROXMOX_HOST}" "pct exec ${FULLRES_CT_ID} -- bash -lc 'hostname; df -h; echo; du -sh /home/uploader/uploads 2>/dev/null || true; du -sh /home/uploader/uploads/jobs 2>/dev/null || true; find /home/uploader/uploads/jobs -type f | wc -l 2>/dev/null || true'"
echo

echo "-- Tunnel CT status + DNS sanity --"
ssh "${PROXMOX_HOST}" "pct exec ${TUNNEL_CT_ID} -- bash -lc 'systemctl is-active cloudflared || true; cat /etc/resolv.conf; getent hosts api.cloudflare.com | head -n 2 || true'"
echo

echo "-- Public endpoint probe --"
curl -fsS --max-time 20 "https://saas.pga.cl/files/jobs/fullres-smoke.txt" >/dev/null \
  && echo "saas.pga.cl: OK" \
  || echo "saas.pga.cl: FAIL"


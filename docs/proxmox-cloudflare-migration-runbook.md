# Proxmox + Cloudflare Migration Runbook

## Current Topology
- Full-res storage container: `CT 118` (`fullres-storage`)
- Full-res files path: `/home/uploader/uploads/jobs/{jobId}/sermon_fullres.mp4`
- Public URL base: `https://saas.pga.cl/files`
- Tunnel runner container: `CT 102` (`nginx`)
- Cloudflare tunnel ID: `dd317cbb-42a0-490e-a09f-71be6e940e1e`
- Ingress for full-res:
  - `hostname: saas.pga.cl`
  - `service: http://192.168.1.19`

## Files/Config To Preserve
- `CT 102:/etc/cloudflared/config.yml`
- `CT 102:/root/.cloudflared/cert.pem`
- `CT 102:/root/.cloudflared/dd317cbb-42a0-490e-a09f-71be6e940e1e.json`
- `CT 118:/etc/nginx/sites-available/fullres.conf`
- `CT 118:/home/uploader/.ssh/authorized_keys`

## DNS/Tunnel Commands
Run on Proxmox host (or any machine with `cloudflared` + cert):

```bash
cloudflared --origincert /root/.cloudflared/cert.pem \
  tunnel route dns --overwrite-dns \
  dd317cbb-42a0-490e-a09f-71be6e940e1e saas.pga.cl
```

Validate DNS:

```bash
dig +short saas.pga.cl A
curl -I https://saas.pga.cl/files/jobs/fullres-smoke.txt
```

## CT 102 DNS Reliability Fix
Cloudflare tunnel stability depends on working DNS inside CT 102.

```bash
# On Proxmox host:
pct set 102 --nameserver 1.1.1.1
pct exec 102 -- bash -lc 'printf "nameserver 1.1.1.1\nnameserver 8.8.8.8\n" > /etc/resolv.conf'
pct exec 102 -- systemctl restart cloudflared
pct exec 102 -- systemctl is-active cloudflared
```

## Worker Env Required
Set on worker runtime:

```env
FULLRES_STORAGE_ENABLED=true
FULLRES_STORAGE_SSH_HOST=192.168.1.19
FULLRES_STORAGE_SSH_USER=uploader
FULLRES_STORAGE_SSH_PORT=22
FULLRES_STORAGE_SSH_PATH=/home/uploader/uploads
FULLRES_STORAGE_PUBLIC_BASE_URL=https://saas.pga.cl/files
FULLRES_STORAGE_SSH_IDENTITY_FILE=/Users/pablogallardo/.ssh/id_ed25519
```

## App Behavior
- Worker uploads full-res output via SSH/SCP to CT 118.
- Worker writes `jobs.metadata.full_res_video_url`.
- `/api/jobs/:id/download` redirects to that URL first.

## Cutover Checklist
1. Bring up replacement CTs (storage + tunnel runner).
2. Restore config files above.
3. Validate SSH upload (`uploader`).
4. Validate public GET from `https://saas.pga.cl/files/...`.
5. Update worker env if CT IP changed.
6. Run one pipeline job and verify dashboard download button.

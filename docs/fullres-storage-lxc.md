# Full-Res Storage LXC

## Provisioned Container
- Proxmox CT: `118`
- Hostname: `fullres-storage`
- IP: `192.168.1.19` (DHCP)
- Services: `nginx`, `openssh-server`
- Upload user: `uploader`
- Files served from: `/home/uploader/uploads`
- Public path on container: `http://192.168.1.19/files/...`
- Cloudflare public URL: `https://saas.pga.cl/files/...` (via existing tunnel in CT `102`)

## Worker Integration
The worker now supports remote full-res upload over SSH/SCP. New env vars:
- `FULLRES_STORAGE_ENABLED`
- `FULLRES_STORAGE_SSH_HOST`
- `FULLRES_STORAGE_SSH_USER`
- `FULLRES_STORAGE_SSH_PORT`
- `FULLRES_STORAGE_SSH_PATH`
- `FULLRES_STORAGE_PUBLIC_BASE_URL`
- `FULLRES_STORAGE_SSH_IDENTITY_FILE`

When enabled, rendered full-res files are uploaded to:
- `${FULLRES_STORAGE_SSH_PATH}/jobs/{jobId}/sermon_fullres.mp4`

And job metadata stores:
- `metadata.full_res_video_url`

## Dashboard Download Behavior
`/api/jobs/:id/download` now:
1. Redirects to `metadata.full_res_video_url` if available.
2. Falls back to local filesystem `metadata.full_res_video_path` behavior.

## Important
Cloudflare tunnel ingress now includes:
- `hostname: saas.pga.cl`
- `service: http://192.168.1.19`

Container `102` DNS was fixed to keep tunnel stable (`nameserver 1.1.1.1`).

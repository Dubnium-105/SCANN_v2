# SCANN Linux Deployment

`docker/` is now a deployment-only layer. It no longer contains a copied backend
or frontend codebase. Images are built directly from:

- `scann_v2/src/scann/native_annotation` for the backend
- `scann_v2/frontend` for the frontend

## Quick Start

1. Copy the repository to the Linux server, or create a bundle with:
   - `powershell -ExecutionPolicy Bypass -File .\docker\package_bundle.ps1`
   - `bash ./docker/package_bundle.sh`
2. On the server, copy `docker/.env.example` to `docker/.env`.
3. Set at least:
   - `SCANN_NATIVE_JWT_SECRET`
   - `SCANN_DATASET_DIR`
   - `FRONTEND_PORT`
4. Put FITS files under:
   - `${SCANN_DATASET_DIR}/new`
   - `${SCANN_DATASET_DIR}/old`
   - `${SCANN_DATASET_DIR}/new_marked`
5. Deploy:

```bash
cd docker
chmod +x deploy.sh
./deploy.sh
```

## Runtime Data

The backend stores all runtime data under the dataset root:

- raw FITS inputs in `new/`, `old/`, `new_marked/`
- `annotations.json`
- `annotation_revisions/`
- `scann_native.db`

## Update Flow

```bash
git pull
cd docker
./deploy.sh
```

## Useful Commands

```bash
cd docker
docker compose --env-file .env ps
docker compose --env-file .env logs -f
docker compose --env-file .env down
```

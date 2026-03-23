# Deployment Notes

## Recommended Linux layout

```text
/srv/scann/
├── app/        # checked-out repository or unpacked deployment bundle
└── dataset/    # FITS files and runtime annotation data
```

Example `.env`:

```dotenv
FRONTEND_PORT=8080
BACKEND_BIND_ADDRESS=127.0.0.1
BACKEND_PORT=8000
SCANN_DATASET_DIR=/srv/scann/dataset
SCANN_NATIVE_JWT_SECRET=replace-this-with-a-long-random-secret
SCANN_NATIVE_JWT_EXPIRE_MINUTES=120
SCANN_NATIVE_TASK_LOCK_TIMEOUT_SECONDS=1200
```

## First deployment

```bash
cd /srv/scann/app/docker
cp .env.example .env
vim .env
mkdir -p /srv/scann/dataset/new /srv/scann/dataset/old /srv/scann/dataset/new_marked
chmod +x deploy.sh
./deploy.sh
```

## Health checks

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8080/health
```

## Re-deploy after code changes

```bash
cd /srv/scann/app
git pull
cd docker
./deploy.sh
```

## Rollback idea

If you deploy from git tags or archived bundles, keep the previous checkout or
previous bundle on disk. Rollback is then:

1. switch back to the previous code snapshot
2. rerun `./docker/deploy.sh`
3. keep the same dataset directory mounted

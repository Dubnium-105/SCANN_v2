# CI/CD Setup

This repository is configured for:

- GitHub-hosted CI on `ubuntu-latest`
- self-hosted CD on the target Linux server

## Pipeline shape

On every PR and every push to `main`, GitHub Actions will:

1. run backend checks for `native_annotation`
2. run frontend tests and a production build
3. verify Docker Compose and both container builds

On pushes to `main`, if all checks pass, GitHub Actions will also:

4. run a deploy job on the Linux server through a self-hosted runner
5. sync the repository into a fixed deployment directory
6. execute `docker/deploy.sh`
7. fail the workflow if backend or frontend health checks do not come up

## Files

- `.github/workflows/pipeline.yml`
- `docker/deploy.sh`
- `docker/deploy_from_runner.sh`

## Recommended production runner labels

Register the self-hosted runner on the Linux server with:

- `self-hosted`
- `linux`
- `scann-prod`

## Server prerequisites

Install at least:

- Docker Engine
- Docker Compose plugin
- `rsync`
- `curl`

Create the runtime directories:

```bash
sudo mkdir -p /srv/scann/app
sudo mkdir -p /srv/scann/dataset/new /srv/scann/dataset/old /srv/scann/dataset/new_marked
sudo chown -R "$USER":"$USER" /srv/scann
```

## GitHub configuration

Create a repository variable or environment variable:

- `DEPLOY_PATH=/srv/scann/app`

Create a protected environment:

- `production`

## First production bootstrap

After the runner is online, do one manual bootstrap on the server:

```bash
cd /srv/scann/app
git clone <your-repo-url> .
cd docker
cp .env.example .env
vim .env
chmod +x deploy.sh
./deploy.sh
```

From that point on, pushes to `main` can redeploy automatically through the
self-hosted runner.

## Notes

- The CI workflow intentionally gates on the stable native backend suite:
  `test_auth`, `test_dataset`, `test_health`, `test_task_locking`.
- You can widen that gate later after any remaining legacy test failures are
  cleaned up.
- The deploy job loads `docker/.env` on the server before running health
  checks, so custom `BACKEND_PORT` and `FRONTEND_PORT` values are respected.

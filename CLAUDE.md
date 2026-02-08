# Minsky RWKV-T5

## Deployment
- Deploy to Lambda training server with `scripts/deploy.sh` (rsync to `nathan-lambda.taila16957.ts.net:/media/external-drive/minsky`)
- After deploy: SSH in, `uv sync`, then `./scripts/launch_experiment.sh`

## TODO
- [ ] Implement proper Python sandbox (e.g. gVisor/nsjail) and re-enable python_exec tool

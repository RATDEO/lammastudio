# Rollback Procedures

Generated: 2024-12-20

## Rollback Methods

### 1. Configuration Rollback (Primary Method)

```bash
# Restore docker-compose.yml
cp artifacts/snapshots/docker-compose.yml.bak docker-compose.yml

# Restore config directory
cp -r artifacts/snapshots/config_*/ config/

# Restore controller code (if modified)
cp -r artifacts/snapshots/controller_*/ controller/
```

### 2. Git-based Rollback

```bash
# Check current changes
git diff HEAD

# Rollback all uncommitted changes
git checkout -- .

# Rollback to specific commit
git reset --hard <commit-hash>
```

### 3. Container Rollback

```bash
# Stop and remove current containers
docker compose down

# Start with previous configuration
docker compose up -d
```

### 4. Service Restart (if using systemd)

```bash
# Restart controller
systemctl restart vllm-controller  # if applicable

# Or restart via start.sh
./start.sh
```

## Pre-Change Checklist

Before any change:
1. [ ] Snapshot exists in artifacts/snapshots/
2. [ ] Git status captured
3. [ ] Rollback command documented in CHANGELOG.md

## Post-Failure Recovery

If a model fails to load after changes:
1. Kill any stuck inference processes: `pkill -f "vllm serve"`
2. Restore previous configuration from snapshots
3. Restart the controller

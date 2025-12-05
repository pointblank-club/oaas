# GitHub Secrets Setup for slashexx/oaas

## Required Secrets

### For `dockerhub-deploy.yml`:
1. **GCP_SERVICE_ACCOUNT_KEY** - GCP service account JSON key for accessing GCP Cloud Storage
2. **DOCKERHUB_USERNAME** - Your Docker Hub username
3. **DOCKERHUB_TOKEN** - Docker Hub access token (not password)

### For `jotai-tests.yml`:
1. **GCP_SERVICE_ACCOUNT_KEY** - Same as above

### For `deploy.yml`:
1. **SSH_HOST** - Server hostname/IP
2. **SSH_USERNAME** - SSH username
3. **SSH_PRIVATE_KEY** - SSH private key
4. **SSH_PORT** - SSH port (optional, defaults to 22)

## How to Copy Secrets from Previous Repo

### Option 1: Manual Copy (Recommended)
1. Go to old repo: `https://github.com/skysingh04/oaas/settings/secrets/actions`
2. For each secret, click on it to view (you can't see the value, but you can verify the name)
3. Go to new repo: `https://github.com/slashexx/oaas/settings/secrets/actions`
4. Click "New repository secret" and add each one

### Option 2: Using GitHub CLI (if you have access)
```bash
# List secrets from old repo (names only, not values)
gh secret list --repo skysingh04/oaas

# Note: You cannot export secret values directly for security reasons
# You'll need to get the actual values from:
# - GCP Console (for GCP_SERVICE_ACCOUNT_KEY)
# - Docker Hub settings (for DOCKERHUB_TOKEN)
# - Your server/SSH config (for SSH secrets)
```

### Option 3: Recreate Secrets
If you can't access the old repo secrets, recreate them:

**GCP_SERVICE_ACCOUNT_KEY:**
1. Go to GCP Console → IAM & Admin → Service Accounts
2. Create or use existing service account
3. Grant "Storage Object Viewer" role for bucket `llvmbins`
4. Create JSON key and copy entire JSON content

**DOCKERHUB_TOKEN:**
1. Go to Docker Hub → Account Settings → Security
2. Create new access token
3. Copy the token (you won't see it again!)

**SSH Secrets:**
- Get from your server configuration or recreate SSH key pair

## Quick Setup Checklist

- [ ] Enable Actions: Settings → Actions → "Allow all actions and reusable workflows"
- [ ] Add `GCP_SERVICE_ACCOUNT_KEY` secret
- [ ] Add `DOCKERHUB_USERNAME` secret  
- [ ] Add `DOCKERHUB_TOKEN` secret
- [ ] Add `SSH_HOST` secret (if using deploy.yml)
- [ ] Add `SSH_USERNAME` secret (if using deploy.yml)
- [ ] Add `SSH_PRIVATE_KEY` secret (if using deploy.yml)
- [ ] Add `SSH_PORT` secret (optional, if using deploy.yml)

## Verify Workflows Will Run

After adding secrets, workflows will automatically run on:
- **dockerhub-deploy.yml**: Push to `main`/`develop` or tags
- **jotai-tests.yml**: Push/PR to any branch
- **deploy.yml**: Manual trigger or after dockerhub-deploy completes


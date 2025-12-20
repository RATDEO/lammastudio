# /models Route 404 Fix

## Investigation Summary

**Issue**: https://app.homelabai.org/models returns 404

**Investigation Date**: 2025-12-20

## Root Cause

The frontend Next.js application did not have a `/models` route defined. The app structure includes:
- `/` - Main dashboard
- `/recipes` - Recipe/model management page
- `/chat` - Chat interface
- `/logs` - Logs viewer
- API routes at `/api/*`

The `/models` route was never created, resulting in a 404 response from Next.js.

## Evidence

### 1. HTTP Headers Analysis

```bash
$ curl -I http://localhost:3000/models
HTTP/1.1 404 Not Found
x-nextjs-cache: HIT
x-nextjs-prerender: 1
X-Powered-By: Next.js
```

The `X-Powered-By: Next.js` header confirms the 404 is from the Next.js application itself, not from a reverse proxy or edge server.

### 2. Frontend Route Structure

```
/home/ser/workspace/projects/lmvllm/frontend/src/app/
├── api/
├── chat/page.tsx
├── logs/page.tsx
├── recipes/page.tsx
└── page.tsx (root)
```

No `/models` directory exists in the app router.

### 3. Backend API Endpoint

The controller API has an OpenAI-compatible `/v1/models` endpoint:

```bash
$ curl http://localhost:8080/v1/models
{"object":"list","data":[...]} # Returns list of available recipes/models
```

This is a backend API endpoint, not a frontend page route.

## Solution

Since "models" and "recipes" are synonymous in vLLM Studio (a recipe is a model configuration), the most user-friendly solution is to redirect `/models` to `/recipes`.

### Implementation

Added a permanent redirect in `frontend/next.config.ts`:

```typescript
async redirects() {
  return [
    {
      source: '/models',
      destination: '/recipes',
      permanent: true,
    },
  ];
}
```

### Verification

After rebuilding and restarting the frontend container:

```bash
$ curl -I http://localhost:3000/models
HTTP/1.1 308 Permanent Redirect
location: /recipes
Refresh: 0;url=/recipes
```

The redirect works correctly with HTTP 308 (Permanent Redirect).

## Files Modified

1. `/home/ser/workspace/projects/lmvllm/frontend/next.config.ts`
   - Added `redirects()` function with `/models -> /recipes` mapping

## Deployment Steps

1. Updated `next.config.ts` with redirect rule
2. Rebuilt frontend Docker image: `docker build -t vllm-studio-frontend:local ./frontend`
3. Stopped old container: `docker stop vllm-studio-frontend && docker rm vllm-studio-frontend`
4. Started new container with updated image: `docker run -d --name vllm-studio-frontend --network host ...`

## Production Deployment

To deploy this fix to production:

1. Push the updated `next.config.ts` to the repository
2. Rebuild the frontend image in CI/CD or manually
3. Push to `ghcr.io/0xsero/vllmstudio/frontend:latest`
4. Restart the frontend container to pull the new image

## Alternative Solutions Considered

1. **Create a `/models` page**: Would duplicate the `/recipes` page functionality
2. **Redirect to root `/`**: Less intuitive - users searching for "models" want to manage them
3. **API proxy route**: Would expose backend API, not a user-facing page

The redirect to `/recipes` is the cleanest solution as it leverages existing functionality without duplication.

## Testing Checklist

- [x] `/models` returns 308 redirect
- [x] `/models` redirects to `/recipes`
- [x] `/recipes` page loads correctly (200 OK)
- [x] Redirect header includes `location: /recipes`
- [ ] Test on production (https://app.homelabai.org/models) after deployment

## Notes

- The redirect is permanent (308) which is SEO-friendly and cacheable
- Next.js redirects work at the framework level, before reaching app routes
- Config changes require a rebuild and restart of the Next.js server

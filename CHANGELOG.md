# Changelog

All notable changes to vLLM Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Frontend: Added redirect from `/models` to `/recipes` to fix 404 error when accessing model management page (2025-12-20)
  - Updated `frontend/next.config.ts` with permanent redirect (HTTP 308)
  - Rebuilt frontend Docker image with the fix

## [0.2.0] - 2024-12-XX

Previous release - vLLM Studio v0.2.0

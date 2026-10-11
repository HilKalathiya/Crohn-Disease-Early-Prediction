@echo off
rem Build the CrohnSight Docker image and test-run it (Docker Desktop must be running).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0docker_tools.ps1" -Mode build

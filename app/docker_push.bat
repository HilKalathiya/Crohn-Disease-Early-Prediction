@echo off
rem Push hilkalathiya59/crohnsight:1.0 to Docker Hub. Create the PRIVATE repo "crohnsight" on hub.docker.com first.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0docker_tools.ps1" -Mode push

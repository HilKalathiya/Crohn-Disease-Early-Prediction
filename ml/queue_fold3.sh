#!/bin/bash
while kill -0 225 2>/dev/null; do sleep 30; done
cd /home/claude && source .venv/bin/activate && python ml/finetune.py effb0_ra 3 10

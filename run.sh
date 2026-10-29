#!/bin/bash
# Ferma ollama se è attivo per liberare la VRAM
sudo systemctl stop ollama 2>/dev/null || pkill -9 -f llama-server 2>/dev/null

# Avvia l'agente
python app.py

# TEO.AI

> DEPLOY. DEFEND. EVOLVE.

## Project Structure

```
Teo.AI/
├── app/
│   ├── backend/
│   │   ├── main.py
│   │   ├── routes/        # API endpoints
│   │   └── core/
│   │       └── teo_engine.py   # model interface — swap here only
│   └── frontend/
│       └── index.html     # pure HTML/CSS/JS
│
├── ai/
│   ├── nanonet/           # NumPy neural net framework
│   ├── rnn/               # RNN layer (upcoming)
│   ├── lstm/              # LSTM layer (next)
│   ├── transformer/       # Transformer (future)
│   └── training/          # training scripts
│
├── datasets/              # training dialogue data
├── docs/
│   └── architecture.md
└── requirements.txt
```

## Quick Start

```bash
cd app/backend
python -m venv .venv && source .venv/bin/activate
pip install -r ../../requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`

## Phase Tracker

- [x] Phase 0 — Architecture (`docs/architecture.md`)
- [x] Phase 1 — Chat pipeline (FastAPI + HTML/JS, placeholder engine)
- [ ] Phase 2 — Memory layer
- [ ] Phase 3 — Project layer
- [ ] Phase 4 — Security audits
- [ ] Phase 5 — Personality (moods, 11-step format, constitution)
- [ ] Phase 6 — Deployment system

## Model Swap Contract

`app/backend/core/teo_engine.py` is the **only** file the app touches.
It exposes one function: `get_response(user_message: str) -> str`
The app never knows or cares what model is behind it.
RNN, LSTM, Transformer — swap here, nothing else changes.

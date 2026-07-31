# Kavach AI — Streamlit Cloud Demo

Kavach AI is a governed semiconductor wafer-map decision-support prototype
evaluated on historical public WM-811K wafer-map data.

## Included

- Frozen 96×96 CNN checkpoint
- Frozen Normal-only auto-clear policy
- 1,000-wafer inference batch
- Decision Ledger with 1,000 decisions
- 205-case engineer review queue
- Drift-aware routing evidence
- Approved good-die occlusion explanation
- Quick wafer-map upload classifier
- Append-only local SQLite review-event chain

## Important limitations

- This is not connected to a live semiconductor fab.
- It does not use proprietary company data.
- Uploaded inputs must be wafer maps, not microscope or SEM photographs.
- Cloud SQLite review events are demonstration state and may reset whenever
  Streamlit Community Cloud restarts or redeploys the app.
- A managed external database is required for durable multi-user audit events.

## Local smoke test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python verify_deployment.py
streamlit run streamlit_app.py
```

## Streamlit Community Cloud

- Repository root: this folder
- Entrypoint: `streamlit_app.py`
- Python: 3.12
- No secrets are required for the current demo package.

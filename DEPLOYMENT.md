# Deployment Steps

1. Create a new GitHub repository.
2. Upload the complete contents of this folder to the repository root.
3. Confirm that `best_model.pt` is below GitHub's ordinary file-size limit.
4. In Streamlit Community Cloud, choose the repository and branch.
5. Set the entrypoint to `streamlit_app.py`.
6. In Advanced settings, select Python 3.12.
7. Deploy and inspect the build logs.
8. Validate:
   - Inference batch = 1,000
   - Auto-cleared = 795
   - Review queue = 205
   - Decision Ledger loads
   - Upload classifier accepts a valid 0/1/2 wafer map
   - Monitoring status = GREEN

This package uses local SQLite only for demonstration. Durable review events
should be migrated to a managed Postgres database in a separate deployment
hardening step.

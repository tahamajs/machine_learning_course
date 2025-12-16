MLflow Server (local)

Start a local MLflow server using docker-compose (creates a local sqlite backend and artifact folder).

1. From the `ThisTerm/final_project` directory run:
   ```bash
   docker compose -f mlflow/docker-compose.yml up -d
   ```
2. Open the MLflow UI at `http://localhost:5000`.
3. Stop the server:
   ```bash
   docker compose -f mlflow/docker-compose.yml down
   ```

Notes
- Artifacts are stored under `ThisTerm/final_project/mlflow/artifacts` by default.
- The compose uses `mlflow` Python package installed at container start; the first start may take a minute.






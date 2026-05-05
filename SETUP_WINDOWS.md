# Running the V3 AlgoTrading Project on Windows

Quick-start guide for cloning the repo on a Windows laptop and bringing the
dashboard + pipeline to life. Tested with PowerShell on Windows 11; should
work on Windows 10 with Python 3.11+ and Node 18+.

---

## 0. Prerequisites

Install once:

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11 or 3.12 | Add to PATH during install. |
| Node.js | 18 LTS or 20 LTS | npm comes with it. |
| Git | latest | for `git clone`. |
| Visual C++ Build Tools | 2022 | Needed by `lightgbm`, `xgboost`, `catboost` wheels on Windows. Install via "Desktop development with C++" workload. |

Optional but recommended:

- VS Code or PyCharm
- Windows Terminal (better than `cmd.exe` for long-running processes)

---

## 1. Clone the repo

```powershell
cd C:\Users\<you>\Documents
git clone <repo-url> AlgoTrading
cd AlgoTrading
```

---

## 2. Python virtual environment + dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

If `requirements.txt` is missing or any wheel fails, the minimum set is:

```powershell
pip install numpy pandas pyarrow yfinance lightgbm xgboost catboost scikit-learn ^
            requests fastapi uvicorn[standard] pydantic python-dotenv pyyaml ^
            sonner openpyxl tensorflow keras
```

(TensorFlow is only needed if you plan to retrain the deep-learning stack.
Fast mode runs without it.)

---

## 3. What's already in the repo (after pull)

| Path | Content | Size |
|---|---|---|
| `V3/01_data/raw/*.parquet` | Cached OHLCV for 100 NSE stocks + global cues + USDINR | ≈ 16 MB |
| `V3/06_results/runs/<run_id>/` | Past pipeline runs — predictions, summaries, plots | ≈ 440 MB |
| `V3/05_live_trading/execution_logs/*.json` | Paper trading transcript JSONs | ≈ 440 KB |
| `V3/05_live_trading/orders/*.json` | Past proposed-order files | small |
| `V3/05_live_trading/ledger/` | Canonical portfolio state | small |
| All Python source under `V3/`, `backend/`, `frontend/src/` | Code | source |
| `RESEARCH_ANALYSIS_LATEST.xlsx` | Thesis-grade comparison workbook | < 1 MB |
| `docs/SESSION_SUMMARY.md`, `docs/LITERATURE_COMPARISON.md` | Thesis-ready docs | source |

**Not in the repo (regenerate locally):**

| Path | Why excluded | How to regenerate |
|---|---|---|
| `V3/02_models/production/<symbol>/` | 3.8 GB — too large for git | Re-run pipeline (step 5 below) |
| `V3/02_models/runs/<run_id>/` | 18 GB | Re-run pipeline |
| `V3/02_models/finbert_india/*.safetensors` | 46 GB | Download from Hugging Face if you want sentiment |
| `V3/01_data/features/` | Regenerated in seconds | `--force-features` flag on first run |
| `venv/`, `node_modules/` | Standard ignores | Step 2 + step 4 |

---

## 4. Frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

---

## 5. First run — regenerate production models

The repo has raw data + past run artefacts but **no model weights**. Run the
pipeline once in fast mode (≈ 10 minutes on a modern laptop) to populate
`V3/02_models/production/`:

```powershell
.\venv\Scripts\Activate.ps1
python V3\07_pipeline\orchestrator.py --fast
```

Wait for `FINAL SUMMARY` to print. You should see something like:

```
Universe: 100  |  trained: 99  |  skipped: 0  |  errors: 0
```

The pipeline writes:

- `V3/06_results/runs/<new_run_id>/` — fresh run dir
- `V3/02_models/production/<symbol>/` — fresh production weights
- `V3/02_models/runs/<new_run_id>/<symbol>/window_*/` — per-window calibration

---

## 6. Start the dashboard

Two terminals.

**Terminal A — backend (FastAPI on :8000):**

```powershell
.\venv\Scripts\Activate.ps1
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal B — frontend (Next.js on :3000):**

```powershell
cd frontend
npm run dev
```

Open: <http://localhost:3000>

---

## 7. Optional — Angel One live credentials

The system supports paper mode (no creds) and live mode (Angel One SmartAPI
creds). For paper-only, skip this step.

For live, create `.env` in the repo root:

```env
ANGEL_API_KEY=your_smartapi_key
ANGEL_CLIENT_ID=AXXXXXXX
ANGEL_PASSWORD=your_pin
ANGEL_TOTP_SECRET=your_base32_totp_secret
TRADING_MODE=paper
```

Verify credentials without placing any orders:

```powershell
python V3\05_live_trading\angel_one_client.py --test
```

The promotion-gate framework refuses to flip `TRADING_MODE=live` until 7
checks pass — see `docs/SESSION_SUMMARY.md` §14.

---

## 8. Daily workflow — research → paper → live

1. **Daily morning:** click **Pipeline → All → Run Pipeline** in the dashboard
   (regenerates predictions for the 100-stock universe).
2. **Live Ops tab:** review predictions, exits-due, ledger NAV.
3. **Robustness tab:** check the cost / turnover / horizon / regime sweeps.
4. **Timing A/B tab:** verify same_close vs next_close vs next_open backtest.
5. **Promotion-gate panel:** track progress towards a paper → live flip.

Long-running tasks (cron) are documented in
`V3/05_live_trading/setup_cron.sh` (Linux) — Windows alternative is Task
Scheduler; example XML to be added.

---

## 9. Common Windows gotchas

- **`venv\Scripts\Activate.ps1` blocked** → `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- **`lightgbm`/`xgboost` install fails** → install Visual C++ Build Tools 2022, then `pip install --no-cache-dir lightgbm xgboost`.
- **TLS / cert errors when fetching Angel master** → handled by `instrument_master.py` via `certifi`. If still failing, run once `python -m pip install --upgrade certifi`.
- **TensorFlow not installing on Python 3.13** → use 3.11 or 3.12.
- **Path length errors** → enable long path support: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -Type DWord` then reboot.

---

## 10. Verifying the install

After step 5 finishes, run:

```powershell
python -c "from fastapi.testclient import TestClient; import importlib.util; s = importlib.util.spec_from_file_location('main', 'backend/main.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); from fastapi.testclient import TestClient; c = TestClient(m.app); print('universe:', c.get('/api/v3/universe').json()['count']); print('audit:', c.get('/api/v3/universe/audit').json())"
```

Expected output:

```
universe: 100
audit: {... 'trained_count': 99, 'skipped_count': 0/1, 'fully_attempted': True, ...}
```

If both numbers are 100 (or 99 trained + 1 skipped for an IPO), the install
is healthy.

---

*Last updated: 2026-05-05.*

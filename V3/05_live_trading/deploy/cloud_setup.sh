#!/usr/bin/env bash
# =============================================================================
# cloud_setup.sh — One-shot deploy for AlgoTrading paper-test on a fresh VM
# =============================================================================
# Tested on Ubuntu 22.04 (AWS Lightsail / EC2 / DigitalOcean / Hetzner).
# Run AS THE NORMAL USER (not root). Uses sudo where needed.
#
# Steps:
#   1.  System packages (python3.11+, git, supervisor or cron, tzdata for IST).
#   2.  Clone the repo into ~/AlgoTrading.
#   3.  Create venv, install requirements.
#   4.  Drop in .env from a secrets file you scp'd over (we do NOT commit it).
#   5.  Smoke-test the pipeline + signal_publisher.
#   6.  Install crontab (evening / morning / reconcile + exit_runner integrated).
#   7.  Set IST timezone so cron firing matches India market hours.
#
# Usage on the VM:
#   curl -fsSL <raw-url-to-this-script> -o cloud_setup.sh
#   bash cloud_setup.sh git@github.com:<you>/AlgoTrading.git ~/.env.algo
#
# Where:
#   $1 = git URL of your private repo (SSH or HTTPS with PAT)
#   $2 = path to a .env file with ANGEL_* + TRADING_MODE=paper (already scp'd)
# =============================================================================
set -euo pipefail

REPO_URL="${1:?Pass repo URL as first arg}"
ENV_FILE="${2:?Pass path to a prepared .env file as second arg}"
INSTALL_DIR="${HOME}/AlgoTrading"

echo "==> 1/7  Installing system packages"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git tzdata cron
sudo ln -fs /usr/share/zoneinfo/Asia/Kolkata /etc/localtime
sudo dpkg-reconfigure -f noninteractive tzdata
echo "    timezone: $(date)"

echo "==> 2/7  Cloning repo to ${INSTALL_DIR}"
if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${INSTALL_DIR}"
else
    git -C "${INSTALL_DIR}" pull --ff-only
fi

echo "==> 3/7  Creating venv & installing requirements"
cd "${INSTALL_DIR}"
python3 -m venv venv
# shellcheck source=/dev/null
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

echo "==> 4/7  Dropping in .env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "    ERROR: ${ENV_FILE} not found." >&2
    echo "    Create it with at least:"
    echo "       ANGEL_API_KEY=..."
    echo "       ANGEL_CLIENT_ID=..."
    echo "       ANGEL_PASSWORD=..."
    echo "       ANGEL_TOTP_SECRET=..."
    echo "       TRADING_MODE=paper"
    exit 1
fi
cp "${ENV_FILE}" "${INSTALL_DIR}/.env"
chmod 600 "${INSTALL_DIR}/.env"

echo "==> 5/7  Smoke test: import pipeline modules"
python3 -c "
import sys; sys.path.insert(0, 'V3/07_pipeline')
from steps.backtest    import run_trade_backtest          # noqa
from steps.diagnostics import run_diagnostics              # noqa
sys.path.insert(0, 'V3/05_live_trading')
import exit_runner, paper_pnl_reconciler, signal_publisher # noqa
print('    all imports OK')
"

echo "==> 6/7  Pre-warming feature cache (one-off, ~5 min)"
# Optional first download — comment out if you've copied caches over
python3 V3/01_data/downloader.py || echo "    (skip if downloader path differs)"

echo "==> 7/7  Installing crontab"
bash V3/05_live_trading/setup_cron.sh
crontab -l

cat <<EOF

==============================================================================
 Deployment finished.

  INSTALL_DIR   : ${INSTALL_DIR}
  Trading mode  : $(grep '^TRADING_MODE' "${INSTALL_DIR}/.env" || echo 'paper (default)')
  Logs          : ${INSTALL_DIR}/V3/05_live_trading/logs/
  Forward-test  : evening 18:00 IST  morning 09:00 IST  reconcile 15:45 IST

 Next:
  1. Verify cron fires: tail -f ${INSTALL_DIR}/V3/05_live_trading/logs/evening.log
  2. After ~10 trading days, exit_runner will start emitting SELLs automatically.
  3. Reconcile mode will produce paper_trading_logs/paper_pnl_<today>.csv daily.

 To flip from paper → live (only when paper looks good for ≥4 weeks):
   sed -i 's/TRADING_MODE=paper/TRADING_MODE=live/' ${INSTALL_DIR}/.env
   bash V3/05_live_trading/setup_cron.sh
==============================================================================
EOF

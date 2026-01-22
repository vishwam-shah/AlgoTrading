# AI Trading System Dashboard

A modern, real-time dashboard for the AI-powered stock trading system built with **Next.js 14**, **FastAPI**, and **TailwindCSS**.

![Dashboard Preview](./docs/dashboard-preview.png)

## Features

- 📊 **Real-time Backtesting** - Run backtests with live progress tracking
- 🎯 **Live Sentiment Analysis** - Google News RSS-powered sentiment using VADER + TextBlob
- 📈 **Interactive Equity Curves** - Recharts-powered visualizations
- 🎨 **Modern Dark UI** - Glass-effect cards with Red Hat Display font
- ⚡ **Fast Performance** - Async background tasks for heavy operations

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Installation

1. **Install Backend Dependencies**
```bash
pip install -r backend/requirements.txt
pip install -r requirements.txt  # Main trading system deps
```

2. **Install Frontend Dependencies**
```bash
cd frontend
npm install
```

3. **Start the Dashboard**
```bash
python start_dashboard.py
```

Or run separately:

```bash
# Terminal 1 - Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

4. **Open Dashboard**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stocks` | GET | List available stocks |
| `/api/sentiment/{symbol}` | GET | Real-time sentiment analysis |
| `/api/backtest` | POST | Start backtest job |
| `/api/backtest/{job_id}` | GET | Check job status |
| `/api/results/{job_id}` | GET | Get completed results |
| `/api/health` | GET | Health check |

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **uvicorn** - ASGI server
- **pydantic** - Data validation

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **TailwindCSS** - Utility-first CSS
- **Recharts** - Chart library
- **Lucide React** - Icon library

### Trading System
- **XGBoost + LightGBM** - Ensemble models
- **VADER + TextBlob** - Sentiment analysis
- **Google News RSS** - Free news data (no rate limits)

## Project Structure

```
AI_IN_STOCK_V2/
├── backend/
│   ├── main.py              # FastAPI server
│   └── requirements.txt     # Backend deps
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── layout.tsx   # Root layout
│   │       ├── page.tsx     # Dashboard page
│   │       └── globals.css  # Global styles
│   ├── package.json
│   └── tailwind.config.js
├── production/
│   ├── orchestrator.py      # Trading orchestrator
│   ├── backtester.py        # Backtest engine
│   └── utils/
│       └── fast_sentiment.py # Google News sentiment
└── start_dashboard.py       # Start both servers
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Optional: Change default ports
BACKEND_PORT=8000
FRONTEND_PORT=3000
```

### Tailwind Theme

The dashboard uses a custom dark theme defined in `frontend/tailwind.config.js`:

- **Background**: `#0a0a0f`
- **Surface**: `#12121a`
- **Primary**: `#6366f1` (Indigo)
- **Accent**: `#22c55e` (Green)

## Usage

1. **Select a Stock** - Choose from the dropdown in the header
2. **Run Backtest** - Click "Run Backtest" to start analysis
3. **View Results** - Watch the equity curve and metrics update in real-time
4. **Check Sentiment** - Live sentiment is displayed in the sidebar

## License

MIT License - see [LICENSE](LICENSE) for details.

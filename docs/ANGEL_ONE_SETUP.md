# Angel One SmartAPI Setup Guide

This guide will help you set up Angel One SmartAPI for live algorithmic trading.

## Prerequisites

1. Angel One trading account
2. Python 3.8+ installed
3. RL trading system set up

## Step 1: Create Angel One Account

If you don't have an Angel One account:
1. Go to https://www.angelone.in/
2. Click "Open Demat Account"
3. Complete KYC verification
4. Fund your account (minimum Rs 10,000 recommended for testing)

## Step 2: Register as SmartAPI Developer

1. Log into your Angel One account
2. Go to SmartAPI portal: https://smartapi.angelbroking.com/
3. Click "Register" or "Sign Up"
4. Fill in developer details
5. Verify email

## Step 3: Create SmartAPI App

1. Log into SmartAPI portal
2. Go to "My Apps" or "Create App"
3. Fill in app details:
   - App Name: AI_Stock_Trading (or any name)
   - App Type: Trading
   - Redirect URL: http://localhost:8080 (for testing)
4. Submit and note your **API Key**

## Step 4: Get TOTP Secret

Angel One requires TOTP (Time-based One-Time Password) for authentication:

1. Go to Angel One app or website
2. Navigate to Settings > Security
3. Enable TOTP authentication
4. Use an authenticator app (Google Authenticator, Authy)
5. **Important**: Note the TOTP secret key shown during setup
   - Usually a 32-character string like: JBSWY3DPEHPK3PXP

## Step 5: Configure Credentials

Create a `.env` file in the project root (this file is gitignored):

```bash
# .env file - DO NOT COMMIT THIS FILE
ANGEL_API_KEY=your_api_key_here
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_trading_password
ANGEL_TOTP_SECRET=your_totp_secret
```

Or create a `broker_config.json` file:

```json
{
    "api_key": "your_api_key_here",
    "client_id": "your_client_id",
    "password": "your_trading_password",
    "totp_secret": "your_totp_secret"
}
```

## Step 6: Install Dependencies

```bash
pip install smartapi-python pyotp
```

## Step 7: Test Connection

```python
from src.rl.brokers.angel_one.api import AngelOneAPI

# Create API instance
api = AngelOneAPI(
    api_key="your_api_key",
    client_id="your_client_id",
    password="your_password",
    totp_secret="your_totp_secret"
)

# Authenticate
if api.authenticate():
    print("Authentication successful!")

    # Get profile
    profile = api.get_profile()
    print(f"Welcome, {profile['data']['name']}")

    # Get funds
    funds = api.get_funds()
    print(f"Available cash: Rs {funds['availablecash']}")
else:
    print("Authentication failed!")
```

## Step 8: Start with Paper Trading

**IMPORTANT**: Always start with paper trading to test your strategies!

```python
from src.rl.brokers.angel_one.api import PaperTradingAPI

# Use paper trading (no real money)
api = PaperTradingAPI(initial_capital=100000)

# Place simulated orders
api.place_market_order('HDFCBANK', 10, 'BUY')

# Check positions
positions = api.get_positions()
print(positions)
```

## API Limits & Best Practices

### Rate Limits
- API calls: 10 requests/second
- Historical data: 1000 candles per request
- WebSocket: 3000 symbols per connection

### Best Practices

1. **Error Handling**
   ```python
   try:
       order_id = api.place_market_order(...)
   except Exception as e:
       logger.error(f"Order failed: {e}")
   ```

2. **Position Verification**
   - Always verify positions after placing orders
   - Use order book to track order status

3. **Risk Controls**
   - Set stop-loss for every trade
   - Use position sizing based on capital
   - Monitor daily P&L limits

4. **Market Hours**
   - NSE: 9:15 AM - 3:30 PM IST
   - Don't place orders outside market hours

5. **Weekend/Holiday Handling**
   - Check trading calendar before placing orders
   - Handle order rejections gracefully

## Troubleshooting

### Authentication Errors

**"Invalid TOTP"**
- Ensure TOTP secret is correct (32 characters)
- Check system time is synchronized
- Generate fresh TOTP before each authentication

**"Invalid API Key"**
- Verify API key from SmartAPI portal
- Check for extra spaces or characters

**"Session Expired"**
- Re-authenticate using `api.authenticate()`
- Sessions expire after market hours

### Order Errors

**"Insufficient Funds"**
- Check available margin in funds API
- Reduce order quantity

**"Invalid Symbol"**
- Use correct symbol format: HDFCBANK-EQ for equity
- Check symbol exists on exchange

**"Market Closed"**
- Verify market hours (9:15 AM - 3:30 PM IST)
- Check for holidays

## Security Recommendations

1. **Never commit credentials** to version control
2. **Use environment variables** for production
3. **Rotate API keys** periodically
4. **Monitor account activity** regularly
5. **Set IP restrictions** in SmartAPI portal if possible
6. **Use paper trading** extensively before live trading

## Support

- Angel One Support: https://www.angelone.in/support
- SmartAPI Documentation: https://smartapi.angelbroking.com/docs
- SmartAPI GitHub: https://github.com/angel-one/smartapi-python

## Next Steps

1. Complete paper trading for at least 2 weeks
2. Analyze paper trading performance
3. Start live trading with small capital (Rs 10,000)
4. Gradually increase capital based on performance

# Energy Insights API 🔋

A FastAPI-based microservice that generates AI-powered weekly energy consumption insights from Snowflake data.

## 🎯 What This Does

Converts the Streamlit dashboard into a hosted API service that:
- Fetches hourly consumption/production data from Snowflake
- Calculates weekly energy scores (0-300 points)
- Generates detailed markdown reports with statistics and anomalies
- Creates AI-powered summaries using OpenAI GPT-4

## 🏗️ Architecture

```
Client Request
     ↓
FastAPI Endpoint (/api/insights)
     ↓
SnowflakeClient → Fetch consumption/production data
     ↓
Insights Engine → Calculate scores, detect anomalies, find opportunities
     ↓
OpenAI GPT-4 → Generate friendly homeowner summary
     ↓
JSON Response with metrics + AI summary
```

## 📦 Project Structure

```
.
├── api.py                      # FastAPI application
├── service/
│   ├── snowflake_client.py     # Data fetching from Snowflake
│   └── insights.py             # Scoring & insights generation
├── test_api.py                 # Test client
├── prompts.py                  # OpenAI prompt templates
├── Dockerfile                  # Container configuration
├── railway.toml                # Railway deployment config
├── requirements-api.txt        # Python dependencies
└── DEPLOYMENT.md               # Detailed deployment guide
```

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements-api.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Run the server:**
   ```bash
   python api.py
   # or
   uvicorn api:app --reload
   ```

4. **Test it:**
   ```bash
   python test_api.py
   ```

5. **View API docs:**
   Open http://localhost:8000/docs

### Deploy to Railway

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete Railway deployment instructions.

**TL;DR:**
1. Push code to GitHub
2. Connect Railway to your repo
3. Add environment variables in Railway dashboard
4. Deploy automatically via Dockerfile

## 🔌 API Usage

### Generate Weekly Insights

**POST** `/api/insights`

```bash
curl -X POST https://your-api.railway.app/api/insights \
  -H "Content-Type: application/json" \
  -d '{
    "site_id": "ABC123",
    "week_start": "2025-10-13",
    "include_ai_summary": true
  }'
```

**Response:**
```json
{
  "site_id": "ABC123",
  "week_start": "2025-10-13",
  "timezone": "America/Chicago",
  "target_kwh": 150.5,
  "current_kwh": 142.3,
  "score": 285,
  "detailed_report": "### 0  Site Information\n...",
  "ai_summary": "Great week! You're 5% under target...",
  "user_info": {
    "email": "user@example.com",
    "full_name": "John Doe"
  },
  "site_info": {
    "address": "123 Main St",
    "city": "Austin",
    "state": "TX",
    "zip": "78701",
    "timezone": "America/Chicago"
  },
  "data_points": 168
}
```

### Get Site Info

**GET** `/api/sites/{site_id}/info`

```bash
curl https://your-api.railway.app/api/sites/ABC123/info
```

### Health Check

**GET** `/health`

```bash
curl https://your-api.railway.app/health
```

## 🧪 Testing

The included test client (`test_api.py`) provides interactive testing:

```bash
python test_api.py
```

It will:
1. Check API health
2. Prompt for a site ID
3. Fetch site information
4. Generate weekly insights
5. Display results

## 💰 Costs

**Railway Free Tier:**
- $5/month credit (usually sufficient for dev/testing)
- Service sleeps after inactivity
- ~500 API calls/month sustainable on free tier

**Per-Request Costs:**
- Snowflake: ~$0.001/request (depends on data volume)
- OpenAI GPT-4-mini: ~$0.001/request
- **Total: ~$0.002 per insight generation**

**Optimization Tips:**
- Set `include_ai_summary: false` to skip OpenAI (50% cost savings)
- Cache results on client side
- Use Railway's sleep-on-idle feature

## 🔐 Security (TODO)

Currently no authentication. To add API key auth:

1. Set `API_KEY` environment variable
2. Add middleware in `api.py`:
   ```python
   from fastapi.security.api_key import APIKeyHeader
   API_KEY_HEADER = APIKeyHeader(name="X-API-Key")
   ```
3. Use in endpoints:
   ```python
   @app.post("/api/insights", dependencies=[Depends(verify_api_key)])
   ```

## 🐛 Troubleshooting

**Connection errors:**
- Verify Snowflake credentials in environment variables
- Check warehouse is running
- Ensure database/schema names match

**OpenAI errors:**
- Verify API key is valid
- Check account has credits
- Review rate limits

**Port conflicts:**
```bash
# Check what's using port 8000
lsof -i :8000
# Use different port
uvicorn api:app --port 8001
```

## 📚 Next Steps

- [ ] Deploy to Railway
- [ ] Test with production data
- [ ] Add API authentication
- [ ] Set up monitoring/alerts
- [ ] Add rate limiting
- [ ] Create webhook notifications
- [ ] Build client SDKs (Python, JS)

## 🤝 Integration Examples

### Python
```python
import requests

response = requests.post(
    "https://your-api.railway.app/api/insights",
    json={"site_id": "ABC123", "week_start": "2025-10-13"}
)
data = response.json()
print(f"Score: {data['score']}/300")
```

### JavaScript
```javascript
const response = await fetch('https://your-api.railway.app/api/insights', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({site_id: 'ABC123', week_start: '2025-10-13'})
});
const data = await response.json();
console.log(`Score: ${data.score}/300`);
```

## 📄 License

Same as parent project.

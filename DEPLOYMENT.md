# Energy Insights API - Deployment Guide

## 🚀 Quick Start (Local Testing)

### 1. Install Dependencies

```bash
pip install -r requirements-api.txt
```

### 2. Configure Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:
```env
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=ENERGY_SHARED
SNOWFLAKE_SCHEMA=TRUNKS_HELIOS
OPENAI_API_KEY=sk-...
```

### 3. Run the API Locally

```bash
# Option 1: Using Python directly
python api.py

# Option 2: Using uvicorn (more control)
uvicorn api:app --reload --port 8000
```

The API will be available at: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

### 4. Test the API

```bash
# In a new terminal
python test_api.py
```

Or use curl:
```bash
# Health check
curl http://localhost:8000/health

# Get site info
curl http://localhost:8000/api/sites/YOUR_SITE_ID/info

# Generate insights
curl -X POST http://localhost:8000/api/insights \
  -H "Content-Type: application/json" \
  -d '{
    "site_id": "YOUR_SITE_ID",
    "week_start": "2025-10-13",
    "include_ai_summary": true
  }'
```

---

## ☁️ Railway Deployment

### Prerequisites
- GitHub account
- Railway account (sign up at https://railway.app with GitHub)

### Step-by-Step Deployment

#### 1. Push Code to GitHub

```bash
# Initialize git repo (if not already done)
git init
git add .
git commit -m "Add FastAPI energy insights service"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

#### 2. Deploy to Railway

1. **Go to Railway Dashboard**
   - Visit https://railway.app
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

2. **Configure Environment Variables**

   In the Railway dashboard, go to your project → Variables tab, and add:

   ```
   SNOWFLAKE_ACCOUNT=your_account
   SNOWFLAKE_USER=your_user
   SNOWFLAKE_PASSWORD=your_password
   SNOWFLAKE_WAREHOUSE=your_warehouse
   SNOWFLAKE_DATABASE=ENERGY_SHARED
   SNOWFLAKE_SCHEMA=TRUNKS_HELIOS
   OPENAI_API_KEY=sk-...
   ```

3. **Deploy**
   - Railway will automatically detect the Dockerfile
   - Click "Deploy" and wait for the build to complete
   - Railway will assign you a public URL like `https://your-service.up.railway.app`

4. **Test Your Deployment**

   ```bash
   # Update API_BASE_URL in test_api.py to your Railway URL
   # Then run:
   python test_api.py
   ```

#### 3. Custom Domain (Optional)

In Railway dashboard:
- Go to Settings → Domains
- Add your custom domain
- Follow DNS configuration instructions

---

## 🐳 Docker Testing (Local)

Test the Docker build locally before deploying:

```bash
# Build the image
docker build -t energy-insights-api .

# Run the container
docker run -p 8000:8000 \
  -e SNOWFLAKE_ACCOUNT=your_account \
  -e SNOWFLAKE_USER=your_user \
  -e SNOWFLAKE_PASSWORD=your_password \
  -e SNOWFLAKE_WAREHOUSE=your_warehouse \
  -e SNOWFLAKE_DATABASE=ENERGY_SHARED \
  -e SNOWFLAKE_SCHEMA=TRUNKS_HELIOS \
  -e OPENAI_API_KEY=sk-... \
  energy-insights-api

# Or use environment file
docker run -p 8000:8000 --env-file .env energy-insights-api
```

---

## 📊 API Endpoints

### `GET /`
Health check - returns service info

### `GET /health`
Detailed health check including Snowflake and OpenAI status

### `GET /api/sites/{site_id}/info`
Get site and user information

**Response:**
```json
{
  "site_id": "ABC123",
  "site_info": {
    "address": "123 Main St",
    "city": "Austin",
    "state": "TX",
    "zip": "78701",
    "timezone": "America/Chicago"
  },
  "user_info": {
    "email": "user@example.com",
    "full_name": "John Doe"
  }
}
```

### `POST /api/insights`
Generate weekly energy insights

**Request:**
```json
{
  "site_id": "ABC123",
  "week_start": "2025-10-13",
  "timezone": "America/Chicago",  // optional
  "include_ai_summary": true      // optional, default: true
}
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
  "ai_summary": "Great week! You're on track...",
  "user_info": {...},
  "site_info": {...},
  "data_points": 168
}
```

---

## 🔒 Security (Optional - Future Enhancement)

To add API key authentication:

1. Add to `.env`:
   ```
   API_KEY=your-secret-key-here
   ```

2. Update `api.py` with authentication middleware:
   ```python
   from fastapi import Security, HTTPException
   from fastapi.security.api_key import APIKeyHeader

   API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

   def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
       if api_key != os.getenv("API_KEY"):
           raise HTTPException(status_code=403, detail="Invalid API key")
       return api_key

   # Then add to endpoints:
   @app.post("/api/insights", dependencies=[Depends(verify_api_key)])
   ```

---

## 🐛 Troubleshooting

### Build fails on Railway
- Check that all required environment variables are set
- Review build logs in Railway dashboard
- Ensure `railway.toml` and `Dockerfile` are committed

### Snowflake connection errors
- Verify credentials are correct
- Check warehouse is running
- Ensure IP whitelisting (if configured) includes Railway IPs

### OpenAI errors
- Verify API key is valid
- Check rate limits on your OpenAI account
- Ensure you have credits available

### Local testing issues
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Try a different port
uvicorn api:app --port 8001
```

---

## 💰 Cost Estimates

**Railway Free Tier:**
- $5 free credit per month
- ~500 hours of uptime
- Service sleeps after inactivity
- Wakes on first request (cold start ~10s)

**Expected Costs:**
- Snowflake: Pay per query (typically pennies per request)
- OpenAI: ~$0.001 per insight with GPT-4-mini
- Railway: Free tier sufficient for <500 requests/month

**Tips to minimize costs:**
- Use `include_ai_summary: false` if AI summary not needed
- Cache results on the client side
- Consider using Railway's "sleep on idle" feature

---

## 📝 Example Integration

### Python Client
```python
import requests

def get_weekly_insights(site_id: str, week_start: str):
    response = requests.post(
        "https://your-service.up.railway.app/api/insights",
        json={
            "site_id": site_id,
            "week_start": week_start,
            "include_ai_summary": True
        }
    )
    return response.json()

# Usage
insights = get_weekly_insights("ABC123", "2025-10-13")
print(f"Score: {insights['score']}/300")
print(f"AI Summary: {insights['ai_summary']}")
```

### JavaScript/Node.js
```javascript
async function getWeeklyInsights(siteId, weekStart) {
  const response = await fetch('https://your-service.up.railway.app/api/insights', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      site_id: siteId,
      week_start: weekStart,
      include_ai_summary: true
    })
  });
  return await response.json();
}

// Usage
const insights = await getWeeklyInsights('ABC123', '2025-10-13');
console.log(`Score: ${insights.score}/300`);
```

---

## 🎯 Next Steps

- [ ] Deploy to Railway
- [ ] Test with real site data
- [ ] Add API key authentication
- [ ] Set up monitoring/logging
- [ ] Add rate limiting
- [ ] Create webhook notifications

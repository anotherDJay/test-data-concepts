# 🚂 Railway Deployment Checklist

Follow these steps to deploy your Energy Insights API to Railway.

## ✅ Pre-Deployment Checklist

- [ ] All code is working locally
- [ ] `.env` file is configured (but NOT committed to git)
- [ ] `.env` is in `.gitignore`
- [ ] `railway.toml` exists and is committed
- [ ] `Dockerfile` exists and is committed
- [ ] `requirements-api.txt` exists with all dependencies
- [ ] You have a GitHub account
- [ ] You have tested the API locally with `python test_api.py`

## 📋 Step-by-Step Deployment Guide

### Step 1: Prepare Your Code Repository

```bash
# Make sure you're in the project directory
cd /Users/djjayalath/Documents/dev/streamlits/bubbles

# Check git status
git status

# Add all new files
git add api.py service/ Dockerfile railway.toml requirements-api.txt .dockerignore DEPLOYMENT.md README-API.md test_api.py .env.example

# Commit
git commit -m "Add FastAPI service for energy insights"

# Push to GitHub (replace with your repo URL)
git push origin main
```

**⚠️ IMPORTANT:** Make sure `.env` is NOT committed! Check with:
```bash
git status | grep .env
# Should only show .env.example, NOT .env
```

---

### Step 2: Create Railway Account

1. Go to https://railway.app
2. Click "Login" or "Start a New Project"
3. Sign in with GitHub
4. Authorize Railway to access your repositories

---

### Step 3: Create New Project on Railway

1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your repository from the list
4. Railway will automatically detect the Dockerfile

---

### Step 4: Configure Environment Variables

In the Railway dashboard:

1. Click on your deployed service
2. Go to the **"Variables"** tab
3. Click **"+ New Variable"** for each of these:

**Required Variables:**

```
SNOWFLAKE_ACCOUNT=<your_snowflake_account>
SNOWFLAKE_USER=<your_snowflake_user>
SNOWFLAKE_PASSWORD=<your_snowflake_password>
SNOWFLAKE_WAREHOUSE=<your_warehouse>
SNOWFLAKE_DATABASE=ENERGY_SHARED
SNOWFLAKE_SCHEMA=TRUNKS_HELIOS
OPENAI_API_KEY=<your_openai_key>
```

**How to find these values:**
- Copy from your local `.env` file
- Or from `.streamlit/secrets.toml`:
  - `SNOWFLAKE_ACCOUNT` → `[snowflake] account`
  - `SNOWFLAKE_USER` → `[snowflake] user`
  - `SNOWFLAKE_PASSWORD` → `[snowflake] password`
  - `SNOWFLAKE_WAREHOUSE` → `[snowflake] warehouse`
  - `OPENAI_API_KEY` → `[openai] api_key`

---

### Step 5: Deploy

1. After adding all environment variables, Railway will **automatically redeploy**
2. Watch the **"Deployments"** tab for build progress
3. Look for:
   - ✅ Build successful
   - ✅ Deploy successful
   - ✅ Health check passing

**Expected build time:** 3-5 minutes

---

### Step 6: Get Your Public URL

1. In Railway dashboard, go to **"Settings"** tab
2. Scroll to **"Networking"** section
3. Click **"Generate Domain"**
4. Copy your public URL (e.g., `https://your-service-name.up.railway.app`)

---

### Step 7: Test Your Deployment

**Option 1: Use the test script**

```bash
# Edit test_api.py
# Change API_BASE_URL = "http://localhost:8000" to:
# API_BASE_URL = "https://your-service.up.railway.app"

python test_api.py
```

**Option 2: Use curl**

```bash
# Health check
curl https://your-service.up.railway.app/health

# Should return:
# {
#   "status": "healthy",
#   "snowflake_connected": true,
#   "openai_configured": true
# }
```

**Option 3: Use browser**

Open in browser:
```
https://your-service.up.railway.app/docs
```

This opens the interactive FastAPI documentation where you can test endpoints directly.

---

### Step 8: Make a Real Request

```bash
curl -X POST https://your-service.up.railway.app/api/insights \
  -H "Content-Type: application/json" \
  -d '{
    "site_id": "YOUR_ACTUAL_SITE_ID",
    "week_start": "2025-10-13",
    "include_ai_summary": true
  }'
```

**Expected response time:**
- First request (cold start): 10-15 seconds
- Subsequent requests: 2-5 seconds

---

## 🐛 Troubleshooting

### Build Fails

**Check:**
- [ ] All files committed and pushed to GitHub
- [ ] Dockerfile syntax is correct
- [ ] requirements-api.txt is present

**Solution:**
- Review build logs in Railway dashboard
- Look for specific error messages
- Ensure all dependencies are in requirements-api.txt

---

### Deployment Succeeds but Health Check Fails

**Check:**
- [ ] All environment variables are set correctly
- [ ] Snowflake credentials are valid
- [ ] OpenAI API key is valid

**Solution:**
1. Go to Railway **"Logs"** tab
2. Look for error messages like:
   - `"Error creating Snowpark session"`
   - `"Failed to connect to Snowflake"`
3. Verify environment variables match your local `.env`

---

### "Snowflake connection not available"

**This means the Snowflake session failed to initialize.**

**Check:**
- [ ] SNOWFLAKE_ACCOUNT is correct (format: `abc12345` or `abc12345.us-east-1`)
- [ ] SNOWFLAKE_USER has correct permissions
- [ ] SNOWFLAKE_PASSWORD is correct (no typos)
- [ ] SNOWFLAKE_WAREHOUSE is running and accessible
- [ ] Database and schema names are correct

**Solution:**
1. Test credentials locally first
2. Check Railway logs for specific Snowflake error
3. Verify warehouse is not suspended

---

### OpenAI Errors

**Check:**
- [ ] OPENAI_API_KEY starts with `sk-`
- [ ] API key is valid and active
- [ ] Account has available credits

**Workaround:**
- Set `"include_ai_summary": false` in requests to skip OpenAI

---

### Service Sleeps After Inactivity

**This is normal on Railway's free tier.**

- Service sleeps after 15 minutes of no requests
- First request after sleep takes ~10 seconds (cold start)
- No data is lost

**To prevent:**
- Upgrade to Railway Pro ($5/month)
- Or set up a cron job to ping the health endpoint every 10 minutes

---

## 💡 Post-Deployment Tips

### Monitor Usage

Railway dashboard shows:
- Request count
- CPU/Memory usage
- Monthly cost estimate
- Logs in real-time

### Custom Domain (Optional)

1. Go to **Settings** → **Domains**
2. Click **"Add Custom Domain"**
3. Enter your domain (e.g., `api.yourdomain.com`)
4. Add DNS records as shown
5. Wait for DNS propagation (5-60 minutes)

### View Logs

```bash
# In Railway dashboard, go to "Logs" tab
# Or use Railway CLI:
railway logs
```

### Update the Service

```bash
# Make code changes locally
git add .
git commit -m "Update API endpoint"
git push origin main

# Railway auto-deploys on push!
```

---

## 📊 Cost Tracking

**Railway Free Tier:**
- $5 credit per month
- Automatically applied
- Tracks usage in dashboard

**What to monitor:**
- Execution time (free tier: ~500 hours/month)
- Outbound data transfer
- Build minutes

**Expected costs for low-traffic API:**
- Railway: $0-2/month (free tier sufficient)
- Snowflake: ~$0.10-1/month (depending on query volume)
- OpenAI: ~$0.50-5/month (depending on insight generation volume)

**Total: ~$0-8/month for dev/testing**

---

## ✅ Success Criteria

You've successfully deployed when:

- [ ] `/health` returns `{"status": "healthy", "snowflake_connected": true}`
- [ ] `/api/sites/{site_id}/info` returns site information
- [ ] `/api/insights` generates a complete report with AI summary
- [ ] Response time is <5 seconds for warm starts
- [ ] No errors in Railway logs

---

## 🎉 You're Done!

Your Energy Insights API is now live at:
```
https://your-service.up.railway.app
```

**Share with your team:**
- API docs: `https://your-service.up.railway.app/docs`
- Health check: `https://your-service.up.railway.app/health`

**Next steps:**
- Integrate with your application
- Set up monitoring/alerts
- Add API authentication
- Create client SDKs

---

## 📞 Need Help?

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- FastAPI Docs: https://fastapi.tiangolo.com
- Check DEPLOYMENT.md for more details

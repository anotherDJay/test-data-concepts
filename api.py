"""FastAPI service for energy insights generation."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import date
from typing import Optional, Dict, Any
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from service.snowflake_client import SnowflakeClient
from service.insights import (
    compute_target,
    score_week,
    compute_insights_report,
    summarize_for_owner
)

# Global client instance
snowflake_client: Optional[SnowflakeClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global snowflake_client

    print("=" * 80)
    print("🚀 Energy Insights API Starting Up")
    print("=" * 80)

    # Check environment variables
    print("\n📋 Environment Check:")
    required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_WAREHOUSE"]
    missing_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask all values for security - only show set/not set status
            print(f"  ✅ {var}: [set]")
        else:
            print(f"  ❌ {var}: NOT SET")
            missing_vars.append(var)

    openai_key = os.getenv("OPENAI_API_KEY")
    print(f"  {'✅' if openai_key else '❌'} OPENAI_API_KEY: {'[set]' if openai_key else 'NOT SET'}")

    # Startup: Initialize Snowflake connection
    connection_params = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "ENERGY_SHARED"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "TRUNKS_HELIOS"),
    }

    print("\n🔌 Connecting to Snowflake...")
    try:
        if missing_vars:
            raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")

        snowflake_client = SnowflakeClient(connection_params)
        print("✅ Snowflake connection established successfully!")
    except Exception as e:
        print(f"\n⚠️  WARNING: Failed to connect to Snowflake")
        print(f"⚠️  Error: {str(e)}")
        print(f"⚠️  Service will start in degraded mode")
        print(f"⚠️  API endpoints will return 503 errors until Snowflake connection is established\n")
        # Don't raise - allow the app to start anyway so healthcheck passes
        snowflake_client = None

    print("=" * 80)
    print("✅ API Server Ready")
    print("=" * 80)

    yield

    # Shutdown: Close Snowflake connection
    if snowflake_client:
        snowflake_client.close()
        print("✅ Snowflake connection closed")


app = FastAPI(
    title="Energy Insights API",
    description="Generate AI-powered energy consumption insights for residential sites",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow all origins for now, tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InsightsRequest(BaseModel):
    """Request model for insights generation."""
    site_id: str = Field(..., description="Helios site ID (e.g., 'ABC123')")
    week_start: date = Field(..., description="Monday date for the week (YYYY-MM-DD)")
    timezone: Optional[str] = Field(None, description="Timezone (e.g., 'America/New_York'). Auto-detected if omitted.")
    include_ai_summary: bool = Field(True, description="Whether to generate AI summary using OpenAI")
    ai_summary_format: str = Field("json", description="Format for AI summary: 'json' for structured data, 'text' for formatted string")


class InsightsResponse(BaseModel):
    """Response model for insights."""
    site_id: str
    week_start: str
    timezone: str
    target_kwh: float
    current_kwh: float
    score: int
    detailed_report: str
    ai_summary: Optional[Dict[str, Any] | str] = None
    token_usage: Optional[Dict[str, Any]] = None
    user_info: Optional[Dict[str, Any]] = None
    site_info: Optional[Dict[str, Any]] = None
    data_points: int


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Energy Insights API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "snowflake_connected": snowflake_client is not None,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }


@app.post("/api/insights", response_model=InsightsResponse)
async def generate_insights(request: InsightsRequest):
    """
    Generate energy insights for a given site and week.

    This endpoint:
    1. Fetches site information and determines timezone
    2. Loads hourly consumption/production data for the specified week
    3. Computes the target based on previous week's performance
    4. Calculates the current week's score
    5. Generates a detailed markdown report
    6. Optionally creates an AI-powered summary for the homeowner

    Returns comprehensive insights including metrics, analysis, and recommendations.
    """
    if not snowflake_client:
        raise HTTPException(status_code=503, detail="Snowflake connection not available")

    try:
        # Get site info and determine timezone
        site_info = snowflake_client.get_site_info(request.site_id)
        if not site_info:
            raise HTTPException(status_code=404, detail=f"Site {request.site_id} not found")

        tz_str = request.timezone or site_info.get("timezone", "UTC")

        # Load current week data
        df_current = snowflake_client.load_week_data(request.site_id, request.week_start, tz_str)
        if df_current is None or df_current.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for site {request.site_id} starting {request.week_start}"
            )

        # Load previous week data to compute target
        from datetime import timedelta
        prev_week_start = request.week_start - timedelta(days=7)
        df_prev = snowflake_client.load_week_data(request.site_id, prev_week_start, tz_str)

        # Compute metrics
        target_kwh = compute_target(df_prev)
        current_kwh = df_current["kwh"].sum()
        score = score_week(current_kwh, target_kwh)

        # Generate detailed report
        detailed_report = compute_insights_report(df_current, site_info, tz_str)

        # Generate AI summary if requested
        ai_summary = None
        token_usage = None
        user_info = None
        if request.include_ai_summary:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")

            user_info = snowflake_client.get_user_info(request.site_id)
            user_name = user_info.get("full_name") if user_info else None

            # Validate format parameter
            if request.ai_summary_format not in ["json", "text"]:
                raise HTTPException(status_code=400, detail="ai_summary_format must be 'json' or 'text'")

            summary_result = summarize_for_owner(
                detailed_report,
                openai_api_key,
                user_name,
                format=request.ai_summary_format
            )

            # Extract content and token usage from result
            ai_summary = summary_result["content"]
            token_usage = summary_result["token_usage"]

        return InsightsResponse(
            site_id=request.site_id,
            week_start=request.week_start.isoformat(),
            timezone=tz_str,
            target_kwh=round(target_kwh, 2),
            current_kwh=round(current_kwh, 2),
            score=score,
            detailed_report=detailed_report,
            ai_summary=ai_summary,
            token_usage=token_usage,
            user_info=user_info,
            site_info=site_info,
            data_points=len(df_current)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")


@app.get("/api/sites/{site_id}/info")
async def get_site_info(site_id: str):
    """Get information about a specific site."""
    if not snowflake_client:
        raise HTTPException(status_code=503, detail="Snowflake connection not available")

    try:
        site_info = snowflake_client.get_site_info(site_id)
        if not site_info:
            raise HTTPException(status_code=404, detail=f"Site {site_id} not found")

        user_info = snowflake_client.get_user_info(site_id)

        return {
            "site_id": site_id,
            "site_info": site_info,
            "user_info": user_info
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching site info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

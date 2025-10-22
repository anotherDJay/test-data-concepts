"""Snowflake data fetching logic extracted from streamlit app."""

import pandas as pd
import re
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col, lit

TF = TimezoneFinder()


class SnowflakeClient:
    """Client for fetching energy data from Snowflake."""

    def __init__(self, connection_params: Dict[str, str]):
        """Initialize Snowflake session with connection parameters."""
        self.session = Session.builder.configs(connection_params).create()

    def get_site_info(self, site_id: str) -> Optional[Dict[str, Any]]:
        """Fetch site address, city, state, zip, and timezone for a given site_id."""
        query = f"""
        SELECT
          p.STREET_ADDRESS AS address,
          p.CITY AS city,
          p.STATE as state,
          p.COORDINATES as coordinates,
          p.ZIP_CODE AS zip
        FROM ENERGY_SHARED.TRUNKS_HELIOS.PROPERTIES p,
             LATERAL FLATTEN(input => p.HELIOS_SITE_IDS) f
        WHERE f.value = '{site_id}'
        LIMIT 1;
        """

        try:
            df = self.session.sql(query).to_pandas()
            df.columns = [c.lower() for c in df.columns]
        except Exception as e:
            raise Exception(f"Error fetching site info: {e}")

        if df.empty:
            return None

        row = df.iloc[0]
        tz = None
        coords = row.get("coordinates")

        if coords is not None:
            lat = lng = None
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                lat, lng = float(coords[0]), float(coords[1])
            elif isinstance(coords, str):
                m = re.search(r"(-?\d+(?:\.\d+)?)[^\d-]+(-?\d+(?:\.\d+)?)", coords)
                if m:
                    lat, lng = float(m.group(1)), float(m.group(2))

            if lat is not None and lng is not None:
                tz = TF.timezone_at(lng=lng, lat=lat)

        return {
            "address": row["address"],
            "city": row["city"],
            "state": row["state"],
            "zip": row["zip"],
            "timezone": tz if tz else "UTC",
        }

    def get_user_info(self, site_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user email and full_name for a given site_id."""
        query = f"""
        SELECT
          u.EMAIL AS email,
          u.FULL_NAME AS full_name
        FROM ENERGY_SHARED.TRUNKS_HELIOS.PROPERTIES p,
             LATERAL FLATTEN(input => p.HELIOS_SITE_IDS) site_flat,
             LATERAL FLATTEN(input => p.HELIOS_USER_IDS) user_flat,
             ENERGY_SHARED.TRUNKS_HELIOS.USERS u
        WHERE site_flat.value = '{site_id}'
          AND u.HELIOS_USER_ID = user_flat.value
        LIMIT 1;
        """

        try:
            df = self.session.sql(query).to_pandas()
            df.columns = [c.lower() for c in df.columns]
        except Exception as e:
            raise Exception(f"Error fetching user info: {e}")

        if df.empty:
            return None

        row = df.iloc[0]
        return {
            "email": row.get("email"),
            "full_name": row.get("full_name"),
        }

    def load_week_data(self, site_id: str, week_start: date, tz_str: str = "UTC") -> Optional[pd.DataFrame]:
        """
        Load one week of consumption + production data for a given site,
        from Monday 00:00 local through Sunday 23:59:59 local.
        """
        # Build local-time window
        local_tz = ZoneInfo(tz_str)
        local_start = datetime.combine(week_start, datetime.min.time()).replace(tzinfo=local_tz)
        local_end = local_start + timedelta(days=7)

        # Convert to naive UTC for Snowflake filters
        start_utc = local_start.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = local_end.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

        try:
            cons = (
                self.session
                .table("ENERGY_SHARED.TRUNKS_HELIOS.HOURLY_CONSUMPTION")
                .filter(
                    (col("HELIOS_SITE_ID") == site_id)
                    & (col("CONSUMPTION_ON") >= lit(start_utc))
                    & (col("CONSUMPTION_ON") < lit(end_utc))
                )
                .select(
                    col("CONSUMPTION_ON").alias("ts"),
                    col("CONSUMPTION_KWH").alias("cons_kwh"),
                )
            )
            prod = (
                self.session
                .table("ENERGY_SHARED.TRUNKS_HELIOS.HOURLY_PRODUCTION")
                .filter(
                    (col("HELIOS_SITE_ID") == site_id)
                    & (col("PRODUCTION_ON") >= lit(start_utc))
                    & (col("PRODUCTION_ON") < lit(end_utc))
                )
                .select(
                    col("PRODUCTION_ON").alias("ts"),
                    col("PRODUCTION_KWH").alias("prod_kwh"),
                )
            )

            joined = cons.join(prod, on="ts", how="full")
            df = joined.to_pandas()
            df.columns = [c.lower() for c in df.columns]
        except Exception as e:
            raise Exception(f"Error loading week data: {e}")

        if df.empty:
            return None

        # Convert back to local tz, fill and compute net
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(tz_str)
        df[["cons_kwh", "prod_kwh"]] = df[["cons_kwh", "prod_kwh"]].fillna(0)
        df["kwh"] = df["cons_kwh"] - df["prod_kwh"]
        df = df.sort_values("ts").reset_index(drop=True)

        return df

    def close(self):
        """Close the Snowflake session."""
        if self.session:
            self.session.close()

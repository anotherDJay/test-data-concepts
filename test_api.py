"""Test client for the Energy Insights API."""

import requests
from datetime import date, timedelta

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change to your deployed URL when testing production


def test_health():
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200


def test_site_info(site_id: str):
    """Test the site info endpoint."""
    print(f"🔍 Testing site info for {site_id}...")
    response = requests.get(f"{API_BASE_URL}/api/sites/{site_id}/info")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Site Info: {data['site_info']}")
        print(f"User Info: {data['user_info']}\n")
    else:
        print(f"Error: {response.json()}\n")
    return response.status_code == 200


def test_insights(site_id: str, week_start: date = None, include_ai: bool = True):
    """Test the insights generation endpoint."""
    if week_start is None:
        # Default to the Monday of last week
        today = date.today()
        days_since_monday = today.weekday()
        last_monday = today - timedelta(days=days_since_monday + 7)
        week_start = last_monday

    print(f"🔍 Testing insights generation for {site_id}, week starting {week_start}...")

    payload = {
        "site_id": site_id,
        "week_start": week_start.isoformat(),
        "include_ai_summary": include_ai
    }

    response = requests.post(f"{API_BASE_URL}/api/insights", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ SUCCESS!")
        print(f"Site ID: {data['site_id']}")
        print(f"Week Start: {data['week_start']}")
        print(f"Timezone: {data['timezone']}")
        print(f"Target: {data['target_kwh']} kWh")
        print(f"Current: {data['current_kwh']} kWh")
        print(f"Score: {data['score']}/300")
        print(f"Data Points: {data['data_points']}")
        print(f"\n--- Detailed Report ---")
        print(data['detailed_report'])
        if data.get('ai_summary'):
            print(f"\n--- AI Summary ---")
            print(data['ai_summary'])
        print("\n")
    else:
        print(f"❌ ERROR: {response.json()}\n")

    return response.status_code == 200


def main():
    """Run all tests."""
    print("=" * 80)
    print("Energy Insights API Test Suite")
    print("=" * 80 + "\n")

    # Test 1: Health check
    if not test_health():
        print("❌ Health check failed. Is the server running?")
        return

    # Test 2: Site info (replace with your actual site ID)
    site_id = input("Enter a site ID to test (or press Enter to skip): ").strip()
    if site_id:
        test_site_info(site_id)

        # Test 3: Insights generation
        week_input = input("Enter week start date (YYYY-MM-DD) or press Enter for last week: ").strip()
        week_start = None
        if week_input:
            try:
                week_start = date.fromisoformat(week_input)
            except ValueError:
                print("Invalid date format, using last week instead")

        include_ai = input("Include AI summary? (y/n, default=y): ").strip().lower() != 'n'
        test_insights(site_id, week_start, include_ai)
    else:
        print("Skipping site-specific tests.")

    print("=" * 80)
    print("Tests complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

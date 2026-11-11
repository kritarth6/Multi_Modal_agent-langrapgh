"""Live Flight & Aviation Tracking tool using Aviationstack API."""

import logging
from typing import Any, Dict, List, Optional
import requests
from src.config import settings

logger = logging.getLogger(__name__)


def track_flights(
    flight_iata: Optional[str] = None,
    airline_name: Optional[str] = None,
    flight_status: Optional[str] = None,
    limit: int = 5,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query real-time flight status, delays, departures, and arrivals via AviationStack API.
    """
    key = api_key or settings.AVIATIONSTACK_API_KEY
    if not key:
        return {
            "success": False,
            "error": "Aviationstack API key not configured.",
            "data": []
        }

    url = "http://api.aviationstack.com/v1/flights"
    params = {
        "access_key": key,
        "limit": limit
    }
    if flight_iata:
        params["flight_iata"] = flight_iata.strip().upper()
    if airline_name:
        params["airline_name"] = airline_name.strip()
    if flight_status:
        params["flight_status"] = flight_status.strip()

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            payload = response.json()
            flights = []
            for item in payload.get("data", []):
                departure = item.get("departure", {}) or {}
                arrival = item.get("arrival", {}) or {}
                airline = item.get("airline", {}) or {}
                flight = item.get("flight", {}) or {}

                flights.append({
                    "flight_number": flight.get("iata") or flight.get("number"),
                    "airline": airline.get("name"),
                    "status": item.get("flight_status", "unknown"),
                    "departure_airport": departure.get("airport"),
                    "departure_iata": departure.get("iata"),
                    "scheduled_departure": departure.get("scheduled"),
                    "estimated_departure": departure.get("estimated"),
                    "arrival_airport": arrival.get("airport"),
                    "arrival_iata": arrival.get("iata"),
                    "scheduled_arrival": arrival.get("scheduled"),
                    "estimated_arrival": arrival.get("estimated"),
                    "delay_minutes": departure.get("delay") or arrival.get("delay") or 0,
                })

            return {
                "success": True,
                "total_records": len(flights),
                "data": flights
            }
        else:
            return {
                "success": False,
                "error": f"Aviationstack returned HTTP {response.status_code}: {response.text}",
                "data": []
            }
    except Exception as e:
        logger.error(f"Flight tracking query failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": []
        }


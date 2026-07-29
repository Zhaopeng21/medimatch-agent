import urllib.parse

import requests
from langchain_core.tools import tool

from app.config.settings import GOOGLE_API_KEY


@tool
def find_local_gp(location: str = "Auckland") -> str:
    """Use Google Places API (New) to find real GP clinics."""
    if not GOOGLE_API_KEY: return "⚠️ Error: Google Maps API Key is missing."
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating"
    }
    payload = {"textQuery": f"medical centre near {location}, Auckland, New Zealand"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10).json()
        places = response.get("places", [])[:3]
        if places:
            res = f"🔚 **Moderate Case: Local Medical Centres Found**\nReal-time data via Google Maps API:\n\n"
            for place in places:
                name = place.get("displayName", {}).get("text", "Unknown Clinic")
                address = place.get("formattedAddress", "No address")
                rating = place.get("rating", "No rating")
                map_url = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(name + ' ' + address)}"
                res += f"📞 **{name}** (⭐{rating})\n📍 *Address:* {address}\n👉 [View on Google Maps]({map_url})\n\n"
            return res
        return f"⚠️ No clinics found near {location} via Google Maps API."
    except Exception as e: return f"⚠️ API Error: {str(e)}"


@tool
def find_urgent_care(location: str = "Auckland") -> str:
    """Use Google Places API (New) to find real emergency centres."""
    if not GOOGLE_API_KEY: return "⚠️ Error: Google Maps API Key is missing."
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress"
    }
    payload = {"textQuery": f"hospital emergency department near {location}, Auckland, New Zealand"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10).json()
        places = response.get("places", [])[:2]
        if places:
            res = f"🔶 **URGENT: Go to ER immediately!**\nReal-time Google Maps hospital data:\n\n"
            for place in places:
                name = place.get("displayName", {}).get("text", "Unknown Hospital")
                address = place.get("formattedAddress", "No address")
                map_url = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(name + ' ' + address)}"
                res += f"🚃 **{name}**\n📍 *Address:* {address}\n👉 [Get Directions]({map_url})\n\n"
            return res + "⚠️ **If it's life-threatening, dial 111 immediately.**"
        return "⚠️ No hospitals found. Call 111."
    except Exception as e: return f"⚠️ API Error: {str(e)}"

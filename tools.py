"""
Tools for EcoHome Energy Advisor Agent
"""
import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager
import requests

# Initialize database manager
db_manager = DatabaseManager()

@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.
    
    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7)
    
    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, and solar irradiance
        E.g:
        forecast = {
            "location": ...,
            "forecast_days": ...,
            "current": {
                "temperature_c": ...,
                "condition": random.choice(["sunny", "partly_cloudy", "cloudy"]),
                "humidity": ...,
                "wind_speed": ...
            },
            "hourly": [
                {
                    "hour": ..., # for hour in range(24)
                    "temperature_c": ...,
                    "condition": ...,
                    "solar_irradiance": ...,
                    "humidity": ...,
                    "wind_speed": ...
                },
            ]
        }
    """
    try:
        days = max(1, min(days, 7))

        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return {"error": "Missing OPENWEATHER_API_KEY environment variable"}

        # Convert location -> latitude/longitude
        geocode_url = "http://api.openweathermap.org/geo/1.0/direct"
        # Normalize location to something OpenWeather understands
        raw_location = location.strip()

        # If user provided "City, ST" (like "San Francisco, CA"), strip the state
        if "," in raw_location:
            city_part = raw_location.split(",")[0].strip()
        else:
            city_part = raw_location

        # Default to US unless you later add more logic
        normalized_query = f"{city_part},US"

        geo_params = {"q": normalized_query, "limit": 1, "appid": api_key}
        resp = requests.get(geocode_url, params=geo_params)

        if resp.status_code != 200:
            return {"error": f"Geocoding API error {resp.status_code}: {resp.text}"}

        geo_res = resp.json()
        if not geo_res:
            return {"error": f"Location not found after normalization: {normalized_query}"}

        lat = geo_res[0]["lat"]
        lon = geo_res[0]["lon"]

        # Fetch weather + hourly solar-friendly data
        weather_url = "https://api.openweathermap.org/data/3.0/onecall"
        weather_params = {
            "lat": lat,
            "lon": lon,
            "exclude": "minutely,alerts",
            "units": "metric",
            "appid": api_key
        }

        weather_data = requests.get(weather_url, params=weather_params).json()

        if "current" not in weather_data:
            return {"error": "Unexpected weather response"}

        # Build structured output
        conditions_map = {
            "Clear": "sunny",
            "Clouds": "partly_cloudy",
            "Rain": "rain",
            "Snow": "snow"
        }

        current_raw = weather_data["current"]
        current = {
            "timestamp": datetime.fromtimestamp(current_raw["dt"]).isoformat(),
            "temperature_c": current_raw["temp"],
            "condition": conditions_map.get(current_raw["weather"][0]["main"], "cloudy"),
            "humidity": current_raw["humidity"],
            "wind_speed": current_raw["wind_speed"]
        }

        # Extract hourly forecast up to requested range
        hourly_forecast = []
        for hour in weather_data["hourly"][: days * 24]:
            condition = conditions_map.get(hour["weather"][0]["main"], "cloudy")

            # Approximate solar irradiance based on cloudiness + daytime
            hour_of_day = datetime.fromtimestamp(hour["dt"]).hour
            if 6 <= hour_of_day <= 18:
                clouds = hour.get("clouds", 50)
                irradiance = max(0, int(900 * (1 - clouds / 100)))
            else:
                irradiance = 0

            hourly_forecast.append({
                "timestamp": datetime.fromtimestamp(hour["dt"]).isoformat(),
                "temperature_c": hour["temp"],
                "condition": condition,
                "solar_irradiance": irradiance,
                "humidity": hour["humidity"],
                "wind_speed": hour["wind_speed"]
            })

        return {
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "forecast_days": days,
            "generated_at": datetime.now().isoformat(),
            "current": current,
            "hourly": hourly_forecast
        }

    except Exception as e:
        return {"error": f"Failed to fetch weather forecast: {str(e)}"}

# Implement get_electricity_prices tool
@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.

    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates.
        Example:
        {
            "date": "2025-11-25",
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "is_weekend": false,
            "base_rate_usd_per_kwh": 0.22,
            "hourly_rates": [
                {
                    "hour": 0,
                    "rate": 0.132,
                    "period": "off_peak",
                    "demand_charge": 0.0
                },
                ...
            ]
        }
    """
    try:
        # Default to today if no date is provided
        if date is None:
            dt = datetime.now()
            date = dt.strftime("%Y-%m-%d")
        else:
            # Validate and parse input date
            dt = datetime.strptime(date, "%Y-%m-%d")

        weekday = dt.weekday()  # 0 = Monday, 6 = Sunday
        is_weekend = weekday >= 5

        # Base rate in USD per kWh (rough mock)
        base_rate = 0.22

        # Weekends usually have cheaper TOU pricing
        weekend_multiplier = 0.9 if is_weekend else 1.0

        hourly_rates = []

        for hour in range(24):
            # Define simple TOU bands:
            # - Off-peak: nights & very early morning
            # - Mid-peak: daytime shoulder
            # - On-peak: late afternoon / early evening
            if hour in [0, 1, 2, 3, 4, 5, 23]:
                period = "off_peak"
                rate = base_rate * 0.6   # cheaper
                demand_charge = 0.0
            elif 6 <= hour <= 15:
                period = "mid_peak"
                rate = base_rate * 1.0   # normal
                demand_charge = 0.0
            elif 16 <= hour <= 21:
                period = "on_peak"
                rate = base_rate * 1.5   # expensive
                demand_charge = 0.10     # extra demand fee per kWh block (mock)
            else:  # hour == 22
                period = "mid_peak"
                rate = base_rate * 1.1
                demand_charge = 0.05

            # Apply weekend discount
            rate *= weekend_multiplier
            demand_charge *= weekend_multiplier

            hourly_rates.append({
                "hour": hour,
                "rate": round(rate, 4),
                "period": period,
                "demand_charge": round(demand_charge, 4),
            })

        return {
            "date": date,
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "is_weekend": is_weekend,
            "base_rate_usd_per_kwh": base_rate,
            "hourly_rates": hourly_rates,
        }

    except Exception as e:
        return {"error": f"Failed to get electricity prices: {str(e)}"}

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")
    
    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        
        if device_type:
            records = [r for r in records if r.device_type == device_type]
        
        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": []
        }
        
        for record in records:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": record.consumption_kwh,
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": record.cost_usd
            })
        
        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}

@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_generation_by_date_range(start_dt, end_dt)
        
        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(sum(r.generation_kwh for r in records) / max(1, (end_dt - start_dt).days), 2),
            "records": []
        }
        
        for record in records:
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": record.generation_kwh,
                "weather_condition": record.weather_condition,
                "temperature_c": record.temperature_c,
                "solar_irradiance": record.solar_irradiance
            })
        
        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}

@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.
    
    Args:
        hours (int): Number of hours to look back (default 24)
    
    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        
        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                "average_weather": "sunny" if generation_records else "unknown"
            }
        }
        
        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1
        
        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)
        
        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}

@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.
    
    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return
    
    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        # Initialize vector store if it doesn't exist
        persist_directory = "data/vectorstore"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        # Load documents if vector store doesn't exist
        if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            # Load documents
            documents = []
            for doc_path in ["data/documents/tip_device_best_practices.txt", "data/documents/tip_energy_savings.txt"]:
                if os.path.exists(doc_path):
                    loader = TextLoader(doc_path)
                    docs = loader.load()
                    documents.extend(docs)
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            # Load existing vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)
        
        results = {
            "query": query,
            "total_results": len(docs),
            "tips": []
        }
        
        for i, doc in enumerate(docs):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low"
            })
        
        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}

@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float, 
                           optimized_usage_kwh: float, price_per_kwh: float = 0.12) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.
    
    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)
    
    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    
    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2)
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]

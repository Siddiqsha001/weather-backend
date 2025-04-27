from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
import requests
from supabase import create_client, Client
import asyncio
from functools import partial

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Retrieve the API key from the .env file
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize Gemini
genai.configure(api_key=google_api_key)

# Weather API configuration
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

async def get_user_profile(user_id):
    try:
        # Get user profile from database
        result = await supabase.from_('profiles').select('*').eq('id', user_id).single()
        return result.data if result else None
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        return None

def get_health_recommendations(weather_data, user_profile):
    recommendations = []
    temp = weather_data["temp"]
    humidity = weather_data["humidity"]
    conditions = weather_data["description"].lower()

    if user_profile:
        # Health conditions
        for condition in user_profile.get('health_conditions', []):
            condition = condition.lower()
            if condition == 'asthma':
                if humidity > 70:
                    recommendations.append("⚠️ High humidity may affect your asthma. Consider staying indoors.")
                if 'rain' in conditions or 'mist' in conditions:
                    recommendations.append("🌧️ Damp conditions may trigger asthma. Keep inhaler handy.")
            elif condition == 'heart condition':
                if temp > 30:
                    recommendations.append("❤️ High temperature may strain your heart. Stay in air-conditioned spaces.")
                elif temp < 5:
                    recommendations.append("❄️ Cold weather can affect blood pressure. Stay warm and avoid overexertion.")
            elif condition == 'diabetes':
                if temp > 30:
                    recommendations.append("📊 Heat can affect blood sugar levels. Check levels more frequently.")
                if humidity > 80:
                    recommendations.append("💧 High humidity can affect insulin absorption. Monitor closely.")

        # Weather sensitivities
        for sensitivity in user_profile.get('weather_sensitivities', []):
            sensitivity = sensitivity.lower()
            if sensitivity == 'cold' and temp < 10:
                recommendations.append("🌡️ Temperature is low and you're sensitive to cold. Bundle up well.")
            elif sensitivity == 'heat' and temp > 28:
                recommendations.append("🌞 Temperature is high and you're heat-sensitive. Stay hydrated and in shade.")
            elif sensitivity == 'humidity' and humidity > 70:
                recommendations.append("💧 High humidity detected. Use dehumidifier indoors if possible.")
            elif sensitivity == 'air quality' and weather_data.get("air_quality", 0) > 100:
                recommendations.append("😷 Poor air quality. Consider wearing a mask outdoors.")

        # Allergies
        for allergy in user_profile.get('allergies', []):
            allergy = allergy.lower()
            if allergy == 'pollen':
                if 'clear' in conditions or 'sunny' in conditions:
                    recommendations.append("🌼 High pollen risk today. Take antihistamines if needed.")
            elif allergy == 'dust':
                if 'windy' in conditions:
                    recommendations.append("💨 Windy conditions may stir up dust. Wear a mask if needed.")

    return recommendations

def get_comprehensive_recommendations(weather_data, user_profile=None):
    temp = weather_data["temp"]
    humidity = weather_data["humidity"]
    conditions = weather_data["description"].lower()
    
    # Initialize recommendation categories
    recommendations = {
        "health_advice": [],
        "activities": [],
        "food_suggestions": [],
        "general_advice": []
    }
    
    # Activity recommendations based on weather
    if temp > 35:
        recommendations["activities"].extend([
            "❌ Avoid strenuous outdoor activities",
            "✅ Indoor swimming",
            "✅ Indoor gym workouts",
            "✅ Mall walking",
            "✅ Indoor sports"
        ])
        recommendations["food_suggestions"].extend([
            "🥤 Drink plenty of water (at least 3-4 liters)",
            "🥗 Light meals with high water content",
            "🍎 Fresh fruits and vegetables",
            "🧂 Electrolyte-rich drinks",
            "❌ Avoid heavy, spicy foods"
        ])
        recommendations["general_advice"].extend([
            "⏰ Plan activities early morning or late evening",
            "👕 Wear light, breathable clothing",
            "🧴 Use sunscreen (SPF 50+)",
            "🕶 Wear sunglasses and a hat"
        ])
    elif 25 <= temp <= 35:
        recommendations["activities"].extend([
            "✅ Swimming",
            "✅ Early morning/late evening walks",
            "✅ Beach activities (with proper protection)",
            "⚠️ Moderate outdoor activities"
        ])
        recommendations["food_suggestions"].extend([
            "🥤 Stay well hydrated",
            "🥗 Fresh salads",
            "🍊 Seasonal fruits",
            "🥤 Sports drinks for outdoor activities"
        ])
    else:
        recommendations["activities"].extend([
            "✅ Most outdoor activities are comfortable",
            "✅ Walking, jogging, cycling",
            "✅ Outdoor sports",
            "✅ Sightseeing"
        ])
        recommendations["food_suggestions"].extend([
            "🥤 Regular water intake",
            "🍽️ Regular balanced meals",
            "🥪 Pack snacks for outdoor activities"
        ])

    # Add health-specific recommendations if user profile exists
    if user_profile:
        health_recs = get_health_recommendations(weather_data, user_profile)
        recommendations["health_advice"].extend(health_recs)

    # Weather condition specific advice
    if 'rain' in conditions:
        recommendations["general_advice"].extend([
            "☔ Carry an umbrella",
            "🧥 Wear waterproof clothing",
            "👟 Wear appropriate footwear"
        ])
    elif 'clear' in conditions:
        recommendations["general_advice"].extend([
            "🕶 Wear sunglasses",
            "🧴 Apply sunscreen",
            "👒 Wear a hat for sun protection"
        ])

    # Air quality based recommendations
    if weather_data.get("air_quality", 0) > 100:
        recommendations["health_advice"].append("😷 Consider wearing a mask due to poor air quality")
        recommendations["activities"] = [act for act in recommendations["activities"] 
                                      if not any(outdoor in act.lower() for outdoor in ['outdoor', 'outside', 'beach'])]

    return recommendations

def format_recommendations(recommendations):
    sections = []
    
    if recommendations["health_advice"]:
        sections.append("🏥 Health Recommendations:")
        sections.extend(recommendations["health_advice"])
        sections.append("")
    
    if recommendations["activities"]:
        sections.append("🏃 Activity Recommendations:")
        sections.extend(recommendations["activities"])
        sections.append("")
    
    if recommendations["food_suggestions"]:
        sections.append("🍽️ Food & Hydration Advice:")
        sections.extend(recommendations["food_suggestions"])
        sections.append("")
    
    if recommendations["general_advice"]:
        sections.append("💡 General Advice:")
        sections.extend(recommendations["general_advice"])
    
    return "\n".join(sections)

def get_weather_data(city):
    try:
        params = {
            'q': city,
            'appid': WEATHER_API_KEY,
            'units': 'metric'
        }
        response = requests.get(WEATHER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        weather = {
            "temp": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        return weather

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return {"error": "Failed to fetch weather data."}
    except Exception as err:
        print(f"Other error occurred: {err}")
        return {"error": "Failed to fetch weather data."}

async def get_weather_with_health_context(city, user_id=None):
    weather_data = get_weather_data(city)
    if "error" in weather_data:
        return str(weather_data)
    
    response_parts = [
        f"🌡️ Current weather in {city}:",
        f"Temperature: {weather_data['temp']}°C",
        f"Conditions: {weather_data['description']}",
        f"Humidity: {weather_data['humidity']}%",
        f"Wind Speed: {weather_data['wind_speed']} m/s",
        ""  # Empty line for spacing
    ]
    
    user_profile = None
    if user_id:
        user_profile = await get_user_profile(user_id)
    
    recommendations = get_comprehensive_recommendations(weather_data, user_profile)
    formatted_recommendations = format_recommendations(recommendations)
    response_parts.append(formatted_recommendations)
    
    return "\n".join(response_parts)

def sync_weather_tool(input_str: str, **kwargs):
    user_id = kwargs.get('user_id')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(get_weather_with_health_context(input_str, user_id))
    loop.close()
    return result

class WeatherAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.tools = [
            Tool(
                name="WeatherFetcher",
                func=sync_weather_tool,
                description="Useful to get the current weather of a given city along with health recommendations."
            )
        ]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="chat-conversational-react-description",
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True
        )

    def run(self, message, user_id=None):
        # Extract just the input message for the agent
        try:
            return self.agent.run(message, callbacks=None)
        except Exception as e:
            print(f"Agent error: {str(e)}")
            return f"I encountered an error: {str(e)}"

# Initialize the agent
agent = WeatherAgent()

@app.route('/chat', methods=['POST'])
def handle_chat_with_agent():
    try:
        data = request.get_json()
        message = data.get('message')
        user_id = data.get('user_id')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Call the agent with just the message
        response = agent.run(message, user_id)

        return jsonify({
            'response': response
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to process your request'
        }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)

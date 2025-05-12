import os
from google import genai
from google.genai import types
import requests

def get_coordinates(city_name):
    endPoint = f"https://nominatim.openstreetmap.org/search?city={city_name}&format=json"
    print("EndPoint: ", endPoint)
    response = requests.get(endPoint, headers={"User-Agent": "Mozilla/5.0"})
    print(response)
    data = response.json()
    print(data)
    for location in data:
        if "India" in location.get("display_name", ""):
            print("Location: ", location)
            latitude = location['lat']
            longitude = location['lon']
            return latitude, longitude

def get_weather(city_name, start_date, end_date):
    print("City Name: ", city_name)
    latitude, longitude = get_coordinates(city_name)
    endPoint = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean"

    print("Latitude: ", latitude)
    print("Longitude: ", longitude)
    print("Start Date: ", start_date)
    print("End Date: ", end_date)
    
    response = requests.get(endPoint)
    data = response.json()
    tempResult = ""
    for i in range(len(data['daily']['temperature_2m_mean'])):
        tempResult += ("Date: " + str(data['daily']['time'][i]) + " Temperature: " + str(data['daily']['temperature_2m_mean'][i]) + "\n")
    return tempResult

get_weather_tool = {
    "name": "get_weather",
    "description": "Get Weather for spocified coordinates on specified dates",
    "parameters": {
        "type": "object",
        "properties": {
            "city_name": {
                "type": "string",
                "description": "Name of city user wants to travel"
            },
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format"
            },
            "end_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format"
            }
        },
        "required": ["city_name", "start_date", "end_date"],
    },
}

import streamlit as st
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
tools = types.Tool(function_declarations=[get_weather_tool])
config = types.GenerateContentConfig(tools=[tools])

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
contents = [
    types.Content(
        role="user", parts=[types.Part(text="I'm Planning to go to a trip to hyderabad. Can you tell me the weather from 10th May 2025 to 14th May 2025? Answer in friendly manner.")]
    )
]

response = client.models.generate_content(
    model="gemini-2.0-flash", config=config, contents=contents
)
print(response)

# Extract tool call details
tool_call = response.candidates[0].content.parts[0].function_call

if tool_call.name == "get_weather":
    result = get_weather(**tool_call.args)
    print(f"Function execution result: {result}")

function_response_part = types.Part.from_function_response(
    name=tool_call.name,
    response={"result": result},
)

# Append function call and result of the function execution to contents
contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)])) # Append the model's function call message
contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response

final_response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=config,
    contents=contents,
)

print(final_response.text)
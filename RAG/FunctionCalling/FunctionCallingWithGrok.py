import os
import json
from openai import OpenAI
import requests
from pydantic import BaseModel, Field
import streamlit as st

class GetWeatherInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

def get_weather(**kwargs):
    input_data = GetWeatherInput(**kwargs)
    latitude = input_data.latitude
    longitude = input_data.longitude
    start_date = input_data.start_date
    end_date = input_data.end_date

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

XAI_API_KEY = os.getenv("X_AI_API_KEY")

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

text_prompt = st.chat_input("Ask me anything about your trip")
messages = []

if text_prompt:
    with st.chat_message("user"):
        st.markdown(text_prompt)
    messages.append(
        {
            "role": "user",
            "content": 
        [{
            "type": "text",
            "text": text_prompt
        }]  
    })

    get_current_temperature_schema = GetWeatherInput.model_json_schema()
    tools = [{
        "type": "function",
        "function": 
        {
            "name": "get_weather",
            "description": "Get Weather for spocified coordinates on specified dates",
            "parameters": get_current_temperature_schema,
        }
    }
    ]

    tools_map = {
        "get_weather": get_weather
    }
    response = client.chat.completions.create(
        model="grok-3-latest",
        messages=messages,
        tools=tools,  # The dictionary of our functions and their parameters
        tool_choice="auto")
    
    print(response.choices[0].message)

    messages.append(response.choices[0].message)

    # Check if there is any tool calls in response body
    # You can also wrap this in a function to make the code cleaner

    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:

            # Get the tool function name and arguments Grok wants to call
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Call one of the tool function defined earlier with arguments
            result = tools_map[function_name](**function_args)

            # Append the result from tool function call to the chat message history,
            # with "role": "tool"
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id  # tool_call.id supplied in Grok's response
                }
            )

    response = client.chat.completions.create(
    model="grok-3-latest",
    messages=messages,
    tools=tools,
    tool_choice="auto"
    )

    print(response.choices[0].message.content)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st.markdown(response.choices[0].message.content)

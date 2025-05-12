import requests
import streamlit as st

def get_weather(latitude, longitude, start_date, end_date):
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

def search_flights(checkInDate, checkOutDate):
    response = requests.get(f"https://test.api.amadeus.com/v2/shopping/hotel-offers?cityCode=DEL&checkInDate={checkInDate}&checkOutDate={checkOutDate}&adults=2")
    data = response.json()
    print(data)
    return "Success"

from anthropic import Anthropic
import json
input_messages = []

client = Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])
text_prompt = st.chat_input("Ask me anything about your trip")

if text_prompt:
    input_messages.append(
        {
            "role": "user",
            "content": 
        [{
            "type": "text",
            "text": text_prompt
        }]  
})

    tools = [{
        "name": "get_weather",
        "description": "Get Weather for spocified coordinates on specified dates",
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"}
            },
            "required": ["latitude", "longitude", "start_date", "end_date"],
        },
    }
]
    with st.chat_message("user"):
        st.markdown(text_prompt)
    # input_messages = [{"role": "user", "content": "I'm Planning to go to a trip to delhi. Can you tell me the weather from 10th May 2025 to 14th May 2025?"}]
    response = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=1024,
                    messages=input_messages,
                    tools=tools,
                    # tool_choice={"type": "tool", "name": "get_weather"}
                )


    def call_function(name, args):
        print("Args: ", args)
        if name == "get_weather":
            return get_weather(**args)
        if name == "search_flights":
            return search_flights(**args)
        
    for tool_call in response.content:
        print(tool_call.type)
        if tool_call.type != "tool_use":
            continue

        name = tool_call.name
        args = tool_call.input

        result = call_function(name, args)
        input_messages.append({     
            "role": "assistant",
            "content":[{
                "type": "tool_use",
                "id": tool_call.id,
                "input": tool_call.input,
                "name": tool_call.name,
            }   ]
        })
        input_messages.append({     
            "role": "user",
            "content":[{
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": str(result)
            }   ]
        })
        with st.chat_message("assistant"):
            with st.spinner("Generating"):      
                for msg in input_messages:
                    print(msg)  
                response_2 = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=1024,
                    messages=input_messages,
                    tools=tools
                )
        st.markdown(response_2.content[0].text)
        print(response_2)
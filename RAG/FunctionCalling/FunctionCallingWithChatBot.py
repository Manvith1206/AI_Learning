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

from openai import OpenAI   
import json
input_messages = []

client = OpenAI(api_key=st.secrets["OPEN_AI_API_KEY"])
text_prompt = st.chat_input("Ask me anything about your trip")

if text_prompt:
    input_messages.append({"role": "user", "content": text_prompt})

    tools = [{
        "type": "function",
        "name": "get_weather",
        "description": "Get current temperature for provided coordinates in celsius.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"}
            },
            "required": ["latitude", "longitude", "start_date", "end_date"],
            "additionalProperties": False
        },
        "strict": True
    }
    ]
    with st.chat_message("user"):
        st.markdown(text_prompt)
    # input_messages = [{"role": "user", "content": "I'm Planning to go to a trip to delhi. Can you tell me the weather from 10th May 2025 to 14th May 2025?"}]
    response = client.responses.create(
                    model="gpt-4.1",
                    input=input_messages,
                    tools=tools,
                    tool_choice={"type": "function", "name": "get_weather"}
                )

    print(response.output)

    def call_function(name, args):
        print("Args: ", args)
        if name == "get_weather":
            return get_weather(**args)
        if name == "search_flights":
            return search_flights(**args)
        
    for tool_call in response.output:
        if tool_call.type != "function_call":
            continue

        name = tool_call.name
        args = json.loads(tool_call.arguments)
        print("Args: ", args)
        result = call_function(name, args)
        input_messages.append(tool_call)  # append model's function call message
        input_messages.append({                               # append result message
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result)
        })
        with st.chat_message("assistant"):
            with st.spinner("Generating"):        
                response_2 = client.responses.create(
                    model="gpt-4.1",
                    input=input_messages,
                    tools=tools
                )
        st.markdown(response_2.output_text)

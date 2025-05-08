import requests

def get_weather_data(latitude, longitude, api_key):
    # Construct the URL with the API key
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}"
    
    # Send the GET request to OpenWeatherMap API
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Extract the data you need from the response
        temperature = data["main"]["temp"]  # Temperature in Kelvin
        humidity = data["main"]["humidity"]
        rain = data.get("rain", {}).get("1h", 0)  # Rainfall in the last hour (in mm)
        weather_description = data["weather"][0]["description"]
        
        # Convert temperature from Kelvin to Celsius
        temperature_celsius = temperature - 273.15
        
        # Return the relevant weather data
        return {
            "temperature_celsius": temperature_celsius,
            "humidity": humidity,
            "rainfall_mm": rain,
            "weather_description": weather_description
        }
    else:
        # If there's an error, print the error status and message for debugging
        print(f"Error: {response.status_code}, {response.text}")
        return None

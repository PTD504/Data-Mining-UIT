import requests

def get_weather_data(latitude, longitude, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rain = data.get("rain", {}).get("1h", 0)
        weather_description = data["weather"][0]["description"]
        temperature_celsius = temperature - 273.15
        return {
            "temperature_celsius": temperature_celsius,
            "humidity": humidity,
            "rainfall_mm": rain,
            "weather_description": weather_description
        }
    else:
        print(f"Error fetching weather data: {response.status_code}, {response.text}")
        return None
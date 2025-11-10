import requests

url = 'http://localhost:9696/predict'

input_data = {
    "humidity": 77,
    "windspeed": 12.9987,
    "temp": 20.21,
    "year": 2012,
    "hour_sin": 0.25881904510252074,
    "hour_cos": 0.9659258262890683,
    "month_sin": 0.4999999999999999,
    "month_cos": 0.8660254037844387,
    ...
    # NO I should enter raw data

}

response = requests.post(url, json=input_data).json()
print(response)
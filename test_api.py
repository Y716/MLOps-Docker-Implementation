import requests
import json

# Define the API endpoint
url = "http://localhost:5000/predict"

# Define the payload
payload = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

# Set the headers
headers = {
    "Content-Type": "application/json"
}

# Send the POST request
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Check the response
if response.status_code == 200:
    print("Prediction:", response.json().get("prediction"))
else:
    print("Error:", response.status_code, response.text)

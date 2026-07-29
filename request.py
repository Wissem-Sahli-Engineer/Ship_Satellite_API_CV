import requests

api_url = "https://ship-satellite-api-cv.onrender.com/classify"

# Send the JSON payload expected by your ImageRequest model
payload = {
    "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR1VZvFROGQy23Bz1axLOjH6BwAoiYV_bvrdJhiAriVkA&s=10"
}

response = requests.post(api_url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
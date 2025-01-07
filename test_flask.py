import requests

# Define the server URL
server_url = "http://127.0.0.1:5000"

# Path to your dataset
dataset_path = r"C:\Users\dayya\Downloads\flowersdata.zip"

# Test /upload-dataset
with open(dataset_path, 'rb') as file:
    response = requests.post(f"{server_url}/upload-dataset", files={"file": file})
print("Upload Dataset Response:", response.json())

# Test /train
response = requests.post(f"{server_url}/train")
print("Train Response:", response.json())

# Test /test
response = requests.post(f"{server_url}/test")
print("Test Response:", response.json())

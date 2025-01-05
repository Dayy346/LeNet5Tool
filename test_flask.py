import requests

# Test /upload-dataset
response = requests.post("http://127.0.0.1:5000/upload-dataset")
print("Upload Dataset Response:", response.json())

# Test /train
response = requests.post("http://127.0.0.1:5000/train")
print("Train Response:", response.json())

# Test /test
response = requests.post("http://127.0.0.1:5000/test")
print("Test Response:", response.json())

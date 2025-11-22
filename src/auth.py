import os
import requests
from dotenv import load_dotenv

load_dotenv()

AUTH_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"

def get_access_token():
    client_id = os.environ.get("OPEN_SKY_CLIENT_ID")
    client_secret = os.environ.get("OPEN_SKY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("Client ID or secret not found in environment variables")

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }

    response = requests.post(AUTH_URL, data=data)

    if response.status_code != 200:
        raise Exception(f"Token request failed: {response.status_code}, {response.text}")

    return response.json()["access_token"]

if __name__ == "__main__":
    print(get_access_token())
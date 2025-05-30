import requests
import os

# === Configuratie ===
PEXELS_API_KEY = "yffriWj8ZmhTym12KDZN9upSuQgl16Aw6GPMTQKEogaDLLEXFgHWXJD0"  # <-- vul hier jouw echte key in
SEARCH_QUERY = "Swiss Alps"
DOWNLOAD_DIR = "downloads"
PER_PAGE = 5  # aantal video's per zoekopdracht

# === Setup map ===
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# === API call ===
headers = {
    "Authorization": PEXELS_API_KEY
}
url = f"https://api.pexels.com/videos/search?query={SEARCH_QUERY}&per_page={PER_PAGE}"

response = requests.get(url, headers=headers)

if response.status_code != 200:
    print("Fout bij ophalen van data:", response.status_code)
    exit()

data = response.json()

# === Download de video’s ===
for i, video in enumerate(data.get("videos", [])):
    video_url = video["video_files"][0]["link"]  # kies de eerste (meestal de hoogste kwaliteit)
    video_response = requests.get(video_url)

    if video_response.status_code == 200:
        filename = f"{DOWNLOAD_DIR}/swiss_alps_{i+1}.mp4"
        with open(filename, "wb") as f:
            f.write(video_response.content)
        print(f"✅ Gedownload: {filename}")
    else:
        print(f"❌ Fout bij downloaden video {i+1}")
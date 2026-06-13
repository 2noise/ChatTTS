import os
import json
import urllib.parse

# आपकी जानकारी
GITHUB_USERNAME = "pavank11213-hub" 
REPO_NAME = "ChatTTS"
BRANCH = "main" 

BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH}/"

data = {}
# फोल्डर का नाम यहाँ 'novels' रखा गया है, इसे पक्का कर लें
novels_dir = "novels"

if os.path.exists(novels_dir):
    for novel in os.listdir(novels_dir):
        novel_path = os.path.join(novels_dir, novel)
        if os.path.isdir(novel_path):
            data[novel] = []
            files = sorted(os.listdir(novel_path))
            for file in files:
                if file.endswith(".txt"):
                    episode_name = file.replace(".txt", "")
                    # पाथ सेट करना
                    txt_path = f"{novels_dir}/{novel}/{file}"
                    mp3_path = f"{novels_dir}/{novel}/audio/{episode_name}.mp3"
                    
                    # URL को सुरक्षित बनाना (spaces और स्पेशल कैरेक्टर्स के लिए)
                    safe_txt_url = urllib.parse.quote(txt_path)
                    safe_mp3_url = urllib.parse.quote(mp3_path)
                    
                    episode_data = {
                        "title": episode_name,
                        "text_url": f"{BASE_URL}{safe_txt_url}",
                        "audio_url": f"{BASE_URL}{safe_mp3_url}"
                    }
                    data[novel].append(episode_data)

# data.json फाइल बनाना
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ data.json successfully created!")
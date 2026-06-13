import os
import re
import sys
import time
import asyncio
import edge_tts

BASE_DIR = "novels"

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def scan_novels():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        print(f"📁 '{BASE_DIR}' फोल्डर नहीं मिला था, इसलिए नया बना दिया गया है।")
        return []
    return [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

# --- नया Edge-TTS फंक्शन ---
async def generate_audio(text, output_file):
    # 'hi-IN-MadhurNeural' पुरुष की आवाज़ है। महिला की आवाज़ के लिए 'hi-IN-SwaraNeural' का उपयोग करें।
    communicate = edge_tts.Communicate(text, "hi-IN-MadhurNeural")
    await communicate.save(output_file)
# ---------------------------

def process_conversion(novel_name):
    novel_path = os.path.join(BASE_DIR, novel_name)
    audio_path = os.path.join(novel_path, "audio")

    if not os.path.exists(audio_path):
        os.makedirs(audio_path)

    txt_files = [f for f in os.listdir(novel_path) if f.endswith(".txt")]
    txt_files.sort(key=natural_sort_key)

    total_txt = len(txt_files)
    total_audio = len([f for f in os.listdir(audio_path) if f.endswith(".mp3")])
    pending = max(0, total_txt - total_audio)

    print("\n" + "="*40)
    print(f"📊 फाइल स्टेटस: '{novel_name}'")
    print(f"कुल चैप्टर्स: {total_txt} | पहले से तैयार: {total_audio} | बाकी: {pending}")
    print("="*40)

    if pending == 0:
        print("\n✨ सभी फाइलें पहले से कनवर्टेड हैं!")
        return

    files_to_convert = []
    for f in txt_files:
        audio_filename = f.replace(".txt", ".mp3")
        if not os.path.exists(os.path.join(audio_path, audio_filename)):
            files_to_convert.append(f)

    print("\n🚀 कन्वर्शन शुरू हो रहा है... (रोकने के लिए Ctrl+C दबाएं)\n")
    
    try:
        for index, filename in enumerate(files_to_convert):
            print(f"🔄 कनवर्ट हो रहा है ({index+1}/{len(files_to_convert)}): {filename} ... ", end="", flush=True)
            
            file_path = os.path.join(novel_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text_content = file.read().strip()
            
            if not text_content:
                print(" [खाली फ़ाइल - छोड़ दिया गया]")
                continue
                
            audio_filename = filename.replace(".txt", ".mp3")
            save_to = os.path.join(audio_path, audio_filename)
            
            try:
                # asyncio का उपयोग करके edge-tts को रन करना
                asyncio.run(generate_audio(text_content, save_to))
                print("✅ हो गया!")
                
                # सर्वर को स्टेबल रखने के लिए हल्का सा 2 सेकंड का ब्रेक
                time.sleep(2) 
                
            except Exception as e:
                print(f"\n❌ फाइल कनवर्ट करने में एरर: {str(e)}")
            
        print("\n🎉 कन्वर्शन सफलतापूर्वक पूरा हुआ!")
        
    except KeyboardInterrupt:
        print("\n\n⛔ कन्वर्शन आपके द्वारा रोक दिया गया है (Stop)।")
    except Exception as e:
        print(f"\n❌ कोई गड़बड़ हुई: {str(e)}")

def main():
    print("="*50)
    print("🌐 टर्मिनल हिंदी उपन्यास ऑडियो कनवर्टर (Edge-TTS Mode)")
    print("="*50)
    
    novels = scan_novels()
    if not novels:
        print("❌ कोई उपन्यास नहीं मिला।")
        sys.exit()
        
    print("\nउपलब्ध उपन्यास:")
    for i, novel in enumerate(novels):
        print(f"  {i+1}. {novel}")
    print(f"  0. बाहर निकलें (Exit)")
        
    try:
        choice = int(input("\nकृपया उपन्यास का नंबर चुनें (जैसे 1 या 2): "))
        if choice == 0:
            sys.exit()
        elif 1 <= choice <= len(novels):
            process_conversion(novels[choice-1])
        else:
            print("❌ गलत नंबर!")
    except ValueError:
        print("❌ कृपया केवल नंबर (अंक) दर्ज करें।")

if __name__ == "__main__":
    main()
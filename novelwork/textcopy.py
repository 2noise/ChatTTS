import os
import sys
import time
import re
import json
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from urllib.parse import urljoin
from seleniumbase import Driver
from sbvirtualdisplay import Display

class NovelScraperCLI:
    def __init__(self):
        self.is_running = False
        self.translator = GoogleTranslator(source='auto', target='hi')
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.settings_file = os.path.join(self.script_dir, "scraper_settings.json")
        self.settings = self.load_settings()

    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        default_novels_path = os.path.join(self.script_dir, "novels")
        return {"last_path": default_novels_path, "history": {}}

    def save_settings(self):
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, indent=4)

    def log(self, message):
        print(message)
        sys.stdout.flush()

    def chunk_text(self, text_lines, limit=4000):
        chunks = []
        current_chunk = ""
        for text in text_lines:
            text = text.strip()
            if not text: continue
            
            if len(current_chunk) + len(text) < limit:
                current_chunk += text + "\n\n"
            else:
                chunks.append(current_chunk)
                current_chunk = text + "\n\n"
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def extract_content(self, soup):
        selectors = [
            ('id', 'chr-content'), ('class', 'chr-c'), ('id', 'chapter-content'),
            ('class', 'chapter-content'), ('id', 'novel-content'), ('class', 'reading-content'),
            ('class', 'text-left'), ('class', 'chapter-container'), ('id', 'content'),
            ('class', 'vung_doc'), ('class', 'txt'), ('div', 'content')
        ]
        
        content_div = None
        for attr, value in selectors:
            content_div = soup.find('div', **{attr: value})
            if content_div: break
            
        if not content_div:
            divs = soup.find_all('div')
            if divs:
                content_div = max(divs, key=lambda d: len(d.find_all('p')))
                
        if not content_div:
            return []

        paragraphs = content_div.find_all('p')
        if paragraphs:
            return [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
        else:
            raw_text = content_div.get_text(separator='\n').split('\n')
            return [t.strip() for t in raw_text if t.strip()]

    def find_next_link(self, soup):
        btn = soup.find('a', id=re.compile(r'(?i)next_chap|next-chap|next'))
        if btn and btn.get('href') and btn['href'] != '#': return btn['href']
        
        btn = soup.find('a', class_=re.compile(r'(?i)btn-next|next_page|next'))
        if btn and btn.get('href') and btn['href'] != '#': return btn['href']
        
        btn = soup.find('a', rel=re.compile(r'(?i)next'))
        if btn and btn.get('href') and btn['href'] != '#': return btn['href']

        for a in soup.find_all('a', href=True):
            if a['href'] == '#': continue
            text = a.get_text(strip=True).lower()
            if 'next' in text or '>>' in text or '▶' in text:
                if 'novel' not in text: 
                    return a['href']
        return None

    def start_scraping(self):
        history = self.settings.get("history", {})
        if history:
            self.log("\n--- Previous Scraping History ---")
            for idx, name in enumerate(history.keys(), 1):
                self.log(f"[{idx}] {name} -> Last URL: {history[name]}")
            self.log("[0] Start a brand new novel\n")
            
            choice = input("Select an option (or press Enter for 0): ").strip()
            if choice and choice != '0' and choice.isdigit() and int(choice) <= len(history):
                novel_name = list(history.keys())[int(choice) - 1]
                current_url = history[novel_name]
                self.log(f"\n▶ Resuming '{novel_name}' from last saved URL: {current_url}")
            else:
                novel_name = input("Enter Novel Name: ").strip()
                current_url = input("Enter Starting Chapter URL: ").strip()
        else:
            novel_name = input("Enter Novel Name: ").strip()
            current_url = input("Enter Starting Chapter URL: ").strip()

        if not novel_name or not current_url:
            self.log("❌ Error: Novel Name and URL cannot be empty!")
            return

        base_path = self.settings.get("last_path", os.path.join(self.script_dir, "novels"))
        novel_folder_path = os.path.join(base_path, novel_name)
        
        if not os.path.exists(novel_folder_path):
            os.makedirs(novel_folder_path)

        self.settings["history"][novel_name] = current_url
        self.save_settings()

        self.is_running = True
        self.log(f"\n🚀 --- Starting Download for '{novel_name}' ---")
        self.log("💡 (To stop the scraper safely, press Ctrl+C once)\n")

        fallback_count = 1
        driver = None
        display = None
        
        try:
            self.log("🛡️ Creating Tiny Virtual Monitor (Memory Saver)...")
            # --- THE FIX: An 800x600 virtual screen uses drastically less memory ---
            display = Display(visible=0, size=(800, 600))
            display.start()

            self.log("🛡️ Booting up 'Headed' but Crash-Proof Chrome engine...")
            # --- THE FIX: We combine the Cloudflare bypass with anti-crash memory flags ---
            driver = Driver(
                uc=True, 
                headless=False, # Must be False to fool Cloudflare
                no_sandbox=True, 
                disable_gpu=True, 
                chromium_arg="--disable-dev-shm-usage --blink-settings=imagesEnabled=false --disable-extensions"
            )
            
            while current_url and self.is_running:
                if current_url.startswith("/"):
                    base_domain = re.match(r'(https?://[^/]+)', self.settings["history"].get(novel_name)).group(1)
                    current_url = urljoin(base_domain, current_url)

                try:
                    self.log(f"🔎 Navigating to page...")
                    driver.get(current_url)
                    
                    # Wait for Cloudflare to process the browser
                    time.sleep(5)
                    
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    
                    if "Just a moment" in soup.text or "Checking your browser" in soup.text or "cloudflare" in soup.text.lower():
                        self.log("⚠️ Cloudflare is inspecting. Waiting 5 more seconds...")
                        time.sleep(5)
                        soup = BeautifulSoup(driver.page_source, 'html.parser')
                        if "Just a moment" in soup.text:
                            self.log("❌ Cloudflare hard-blocked this Datacenter IP. Try a different novel site.")
                            break
                    
                    title_tag = soup.find('span', class_='chr-text') or soup.find('h2') or soup.find('h1')
                    english_title = title_tag.text.strip() if title_tag else f"Chapter {fallback_count}"
                    
                    chapter_match = re.search(r'(?:chapter|ch|episode|ep)\D*?(\d+)', english_title.lower())
                    if chapter_match:
                        display_num = chapter_match.group(1)
                        fallback_count = int(display_num)
                    else:
                        display_num = str(fallback_count)

                    hindi_title = self.translator.translate(english_title)
                    self.log(f"⏳ Processing Chapter {display_num}: {english_title[:30]}...")

                    text_lines = self.extract_content(soup)

                    if not text_lines or len(text_lines) < 3:
                        self.log("❌ Error: Web structure blocked or content text not found.")
                        self.log(f"URL Failed: {current_url}")
                        break

                    chunks = self.chunk_text(text_lines)
                    hindi_text = ""
                    
                    for chunk in chunks:
                        if not self.is_running: break
                        try:
                            translated_chunk = self.translator.translate(chunk)
                            if translated_chunk: hindi_text += translated_chunk + "\n\n"
                            time.sleep(1.5) 
                        except Exception as e:
                            self.log(f"⚠️ Translation Error: {e}")
                            hindi_text += chunk + "\n\n" 
                    
                    if not self.is_running: break

                    safe_title = re.sub(r'[\\/*?:"<>|]', "", hindi_title)[:50].strip()
                    filename = os.path.join(novel_folder_path, f"Chapter_{display_num}_{safe_title}.txt")
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(hindi_title + "\n\n" + hindi_text)

                    self.log(f"✅ Saved Successfully: Chapter {display_num}")

                    next_href = self.find_next_link(soup)

                    if next_href:
                        current_url = urljoin(current_url, next_href) 
                        self.settings["history"][novel_name] = current_url
                        self.save_settings()
                        fallback_count += 1 
                        time.sleep(2) 
                    else:
                        self.log("🎉 Finished! Could not find a 'Next' button. All available chapters downloaded.")
                        break

                except Exception as e:
                    self.log(f"❌ Loop/Network Error: {e}")
                    break

        except KeyboardInterrupt:
            self.log("\n🛑 Stopping scraper safely as per request...")
        except Exception as e:
            self.log(f"❌ Fatal Error: {e}")
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
            if display:
                try:
                    display.stop()
                except:
                    pass
            self.log("🏁 Process Ended. Current status has been successfully stored.")

if __name__ == "__main__":
    app = NovelScraperCLI()
    try:
        app.start_scraping()
    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
import os
import sys
import ssl
import urllib.request

# Globally bypass SSL verification for all python libraries (crucial for Windows environments)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["CURL_CA_BUNDLE"] = ""
except Exception:
    pass

def download_via_github_mirror(limit):
    """Downloads raw DailyDialog txt directly from GitHub and parses it."""
    # Correct URL discovered through GitHub API checks
    url = "https://raw.githubusercontent.com/snakeztc/NeuralDialog-LAED/master/data/daily_dialog/train/dialogues.txt"
    print("Attempting direct download from GitHub mirror...")
    print(f"URL: {url}")
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            content = response.read().decode('utf-8')
            
        lines = content.splitlines()
        total_available = len(lines)
        print(f"Successfully loaded raw file. Found {total_available} conversations.")
        
        # Apply the conversation limit
        limit = min(limit, total_available)
        formatted_lines = []
        
        for i in range(limit):
            line = lines[i].strip()
            if not line:
                continue
                
            # Split utterances using the '__eou__' separator
            utterances = [u.strip() for u in line.split("__eou__") if u.strip()]
            
            # Format pairs: User -> AI
            for j in range(0, len(utterances) - 1, 2):
                user_speech = utterances[j]
                ai_speech = utterances[j+1]
                
                # Basic cleanup
                user_speech = " ".join(user_speech.split())
                ai_speech = " ".join(ai_speech.split())
                
                if user_speech and ai_speech:
                    formatted_lines.append(f"User: {user_speech}\n")
                    formatted_lines.append(f"AI: {ai_speech}\n")
                    
        return formatted_lines
    except Exception as e:
        print(f"GitHub mirror download failed: {e}")
        return None

def download_via_huggingface(limit):
    """Fallback method using Hugging Face datasets library with SSL bypassed."""
    print("\nFalling back to Hugging Face 'datasets' library...")
    
    try:
        try:
            import datasets
        except ImportError:
            print("Installing 'datasets' library...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "requests", "tqdm"])
            import datasets
            
        print("Loading 'daily_dialog' dataset...")
        dataset = datasets.load_dataset("daily_dialog", trust_remote_code=True)
        train_data = dataset["train"]
        
        total_available = len(train_data)
        limit = min(limit, total_available)
        formatted_lines = []
        
        for i in range(limit):
            dialogue_turns = train_data[i]["dialog"]
            for j in range(0, len(dialogue_turns) - 1, 2):
                user_speech = dialogue_turns[j].strip()
                ai_speech = dialogue_turns[j+1].strip()
                
                user_speech = " ".join(user_speech.split())
                ai_speech = " ".join(ai_speech.split())
                
                if user_speech and ai_speech:
                    formatted_lines.append(f"User: {user_speech}\n")
                    formatted_lines.append(f"AI: {ai_speech}\n")
                    
        return formatted_lines
    except Exception as e:
        print(f"Hugging Face download failed: {e}")
        return None

def main():
    print("==================================================")
    print("     DailyDialog Conversational Dataset Loader    ")
    print("==================================================")
    
    default_limit = 2000
    print(f"\nHow many dialogues would you like to download?")
    print(f"Default is {default_limit} (or press Enter for default):")
    
    user_input = input("> ").strip()
    if user_input.isdigit():
        limit = int(user_input)
    else:
        limit = default_limit
        
    # Step 1: Try GitHub Mirror
    formatted_lines = download_via_github_mirror(limit)
    
    # Step 2: Fallback to Hugging Face
    if not formatted_lines:
        formatted_lines = download_via_huggingface(limit)
        
    if formatted_lines:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "dataset.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(formatted_lines)
            
        print("\n==================================================")
        print("SUCCESS!")
        print(f"Saved {len(formatted_lines)//2} conversational turns to:")
        print(f"dataset.txt")
        print("==================================================")
    else:
        print("\nFAILED: Unable to download dataset via any method.")

if __name__ == "__main__":
    main()

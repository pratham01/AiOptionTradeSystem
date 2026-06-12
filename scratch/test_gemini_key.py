import requests

def test_model(model):
    api_key = "AIzaSyBz8nrmtgHXuMjRBwbSrcP0qh2aS5_JN4A"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": "Hello, respond with 'System Online' if you can read this."}]}]}
    
    print(f"Testing {model}...")
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"  Status code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            text = data['candidates'][0]['content']['parts'][0]['text']
            print(f"  Success response: {text.strip()}")
            return True
        else:
            print(f"  Error: {response.text.strip()}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    models = ["gemini-1.5-flash", "gemini-1.5-pro-latest", "gemini-2.5-flash", "gemini-1.5-pro"]
    for m in models:
        if test_model(m):
            print(f"\n🎉 Model {m} works successfully!")
            break

if __name__ == "__main__":
    main()

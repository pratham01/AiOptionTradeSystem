from openai import OpenAI
import time
import sys

print("Starting test...")
start_time = time.time()
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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",
  timeout=15.0 # Set a 15-second timeout so it doesn't hang forever
)
try:
    print("Making request...")
    completion = client.chat.completions.create(
      model="minimaxai/minimax-m2.7",
      messages=[{"role":"user","content":"say hello"}],
      temperature=1,
      top_p=0.95,
      max_tokens=50,
      stream=True
    )
    print("Request successful. Reading stream...")
    result = ""
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="")
            sys.stdout.flush()
    print(f"\n\nFinal Result: {result}")
except Exception as e:
    print(f"\nError after {time.time() - start_time:.2f} seconds: {e}")

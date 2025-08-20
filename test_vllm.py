import asyncio
import httpx
import time

API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "Qwen3-Coder-30B-Instruct"

# Generate a long prompt (repeat a snippet to simulate large file)
BASE_SNIPPET = "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number\"\"\"\n"
PROMPT_REPEAT = 1024  # adjust to get desired token length
LONG_PROMPT = BASE_SNIPPET * PROMPT_REPEAT

# Max tokens per request
MAX_TOKENS = 1024  # keep under VRAM limits
# Max concurrency (will auto-adjust if OOM occurs)
MAX_CONCURRENCY = 8
# Iterations per worker
ITERATIONS_PER_WORKER = 5

async def fetch_tokens(client, prompt, max_tokens):
    response = await client.post(API_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0
    })
    data = response.json()
    return data.get("usage", {}).get("completion_tokens", len(data["choices"][0]["text"].split()))

async def worker(name, prompt, max_tokens, iterations):
    async with httpx.AsyncClient(timeout=None) as client:
        total_tokens = 0
        total_time = 0.0
        for i in range(iterations):
            start = time.time()
            try:
                tokens = await fetch_tokens(client, prompt, max_tokens)
            except Exception as e:
                print(f"[Worker {name}] Iteration {i+1} failed: {e}")
                break
            elapsed = time.time() - start
            total_tokens += tokens
            total_time += elapsed
            print(f"[Worker {name}] Iteration {i+1}: {tokens} tokens in {elapsed:.2f}s "
                  f"({tokens/elapsed:.2f} tokens/sec)")
        return total_tokens, total_time

async def run_benchmark(prompt, max_tokens, concurrency, iterations):
    tasks = [worker(i+1, prompt, max_tokens, iterations) for i in range(concurrency)]
    results = await asyncio.gather(*tasks)
    
    total_tokens = sum(r[0] for r in results)
    total_time = sum(r[1] for r in results)
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print(f"\n=== Overall Average Tokens/sec: {avg_tps:.2f} ===")

if __name__ == "__main__":
    print("Starting stress test on Qwen3-Coder-30B...")
    asyncio.run(run_benchmark(LONG_PROMPT, MAX_TOKENS, MAX_CONCURRENCY, ITERATIONS_PER_WORKER))

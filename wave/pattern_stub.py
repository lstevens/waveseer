from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/match")
async def match(request: Request):
    # Consume incoming payload but ignore content, return dummy pattern for demo
    await request.json()
    import random
    # Generate a random pattern ID to make the demo more interesting
    patterns = [
        {"pattern_id": "double-bottom", "score": 0.78},
        {"pattern_id": "head-shoulders", "score": 0.82},
        {"pattern_id": "cup-handle", "score": 0.65},
        {"pattern_id": "flag-pattern", "score": 0.91},
        {"pattern_id": "triangle-ascending", "score": 0.74}
    ]
    return random.choice(patterns)

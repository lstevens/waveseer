from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/match")
async def match(request: Request):
    data = await request.json()
    return data

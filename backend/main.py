from fastapi import FastAPI

app = FastAPI()
@app.post('/analyze')
async def hello():
    return {"message":"hello world"}


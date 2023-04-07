from service.model.model import TextRecognizer
from fastapi import FastAPI
from mangum import Mangum
import asyncio



app = FastAPI()
model = TextRecognizer('model_path')



@app.get("/")
async def get():
    return {
        'message': 'Hello World! Welcome to the (experimental) image recognition API'
    }
    
    
handler = Mangum(app)

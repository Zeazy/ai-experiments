from fastapi import FastAPI
from mangum import Mangum
import asyncio


app = FastAPI()



@app.get("/")
async def get():
    return {
        'message': 'Hello World! Welcome to the (experimental) image recognition API'
    }
    
    



handler = Mangum(app)






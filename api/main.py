from fastapi import FastAPI, Request, Response

app = FastAPI()


@app.get('/')
async def index():
    return {'message': 'Hello World'}

@app.get('/a')
async def a():
    return {'message': 'Api'}
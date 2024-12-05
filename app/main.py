import aioredis
from fastapi import FastAPI, Request, Depends, Response
from starlette.responses import HTMLResponse, JSONResponse
from starlette.templating import Jinja2Templates

from messages.dao import ScoresDAO
from users.router import router as router_users, templates
from fastapi.staticfiles import StaticFiles
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import aioredis
import requests
import uvicorn
from messages.schemas import Score
from users.models import User
from users.dependencies import get_current_user
import json
from random import randint
import pandas as pd

app = FastAPI()


app.mount('/static', StaticFiles(directory='static'), 'static')
templates = Jinja2Templates(directory='templates')

@app.on_event("startup")
async def startup():
    redis = await aioredis.from_url("redis://redis:6379")
    await FastAPILimiter.init(redis)

@app.head('/')
async def head_handler():
    # Возвращаем только метаданные без тела ответа
    return JSONResponse(headers={"Content-Type": "text/html"}, status_code=200)

@app.get('/', dependencies=[Depends(RateLimiter(times=5, seconds=1))])
async def index(request: Request):
    conv_id = requests.post('http://api:8001/new_conversation?api_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiZXhwIjoxNzM1OTcwMTI0fQ.pyRntCTCRnGJM1t9wafVwBtiSGGOULGAhNRioLIY6aI&db_token=MS05ZUl0ZktDLXpGeGVOTFNyZE5uZmFBPT0')

    conv_id = conv_id.json()['conv_id']
    return templates.TemplateResponse('main_page.html', context={'request': request})

@app.post('/scores')
async def scores(score: Score):
    await ScoresDAO.add(**(score.dict()))

@app.get('/dialog/{conv_id}')
async def dialog(conv_id:int, request: Request, user_data: User = Depends(get_current_user)):
    df = pd.read_csv('dataset_with_captions.csv')
    ind = randint(0, len(list(pd['images'])))
    look_img = list(pd['images'])[ind]
    prod_img = list(pd['items_images'])[ind]
    return templates.TemplateResponse('dialog.html', context={'request': request, 'username':user_data.email, 'conv_id': conv_id, 'look': {
        'image': look_img, 'similar': prod_img}})

app.include_router(router_users)







#uvicorn.run('main:app', host="0.0.0.0", port=8000, log_level="info")

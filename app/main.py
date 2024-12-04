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
    conv_id = requests.post('http://api:8001/new_conversation?api_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiZXhwIjoxNzM1OTAwNTY5fQ.xTojJvrfusApuHQzkK8fCw-WCNgYexnerYlVJ0a1bis&db_token=MS14eGZfM29yalVnY2VqbU5DTmVabjlRPT0')

    conv_id = conv_id.json()['conv_id']
    return templates.TemplateResponse('main_page.html', context={'request': request})

@app.get('/scores')
async def scores(score: Score):
    await ScoresDAO.add(**(score.dict()))

@app.get('/dialog')
async def dialog(conv_id:int, request: Request, user_data: User = Depends(get_current_user)):
    return templates.TemplateResponse('dialog.html', context={'request': request, 'username':user_data.email, 'conv_id': conv_id})

app.include_router(router_users)







#uvicorn.run('main:app', host="0.0.0.0", port=8000, log_level="info")

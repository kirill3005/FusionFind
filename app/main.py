import aioredis
from fastapi import FastAPI, Request, Depends, Response
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from users.router import router as router_users, templates
from fastapi.staticfiles import StaticFiles
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import aioredis
import requests
import uvicorn

app = FastAPI()


app.mount('/static', StaticFiles(directory='static'), 'static')
templates = Jinja2Templates(directory='templates')

@app.on_event("startup")
async def startup():
    redis = await aioredis.from_url("redis://redis:6379")
    await FastAPILimiter.init(redis)



@app.get('/', dependencies=[Depends(RateLimiter(times=5, seconds=1))])
async def index(request: Request):
    conv_id = requests.post('http://api.fusionfind.ru/message?apitoken=kirill&db_token=kirill')
    conv_id = conv_id.json()['conv_id']
    return templates.TemplateResponse('main_page.html', context={'request': request, 'conv_id':conv_id})

app.include_router(router_users)





#uvicorn.run('main:app', host="0.0.0.0", port=8000, log_level="info")

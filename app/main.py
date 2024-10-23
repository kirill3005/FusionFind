from fastapi import FastAPI
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from users.router import router as router_users, templates
from fastapi.staticfiles import StaticFiles
from router import router as main_router
from bots.router import router as bot_router
import uvicorn

app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), 'static')
templates = Jinja2Templates(directory='templates')

@app.get('/')
async def index():
    return {'message':'Hello'}

app.include_router(router_users)
app.include_router(main_router)

app.include_router(bot_router)


#uvicorn.run('main:app', host="0.0.0.0", port=8000, log_level="info")

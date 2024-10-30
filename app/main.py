from fastapi import FastAPI, Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from users.router import router as router_users, templates
from fastapi.staticfiles import StaticFiles

import uvicorn

app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), 'static')
templates = Jinja2Templates(directory='templates')

@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse('main_page.html', context={'request': request})

app.include_router(router_users)





#uvicorn.run('main:app', host="0.0.0.0", port=8000, log_level="info")

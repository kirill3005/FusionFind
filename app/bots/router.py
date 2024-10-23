from fastapi import APIRouter, HTTPException, status, Response, Depends, UploadFile, Request, File
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse

from bots.dao import BotDAO
from bots.schemas import NewBot

from users.models import User
from users.dependencies import get_current_user


router = APIRouter(prefix='/bots', tags=['Работа с пользователями'])
templates = Jinja2Templates(directory='templates')

@router.post('/new')
async def new(user_data: User = Depends(get_current_user)):
    await BotDAO.add(**{'user_id': user_data.id})


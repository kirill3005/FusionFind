from datetime import datetime

from fastapi import APIRouter, Request, Depends
from starlette.templating import Jinja2Templates
from datetime import date, datetime
from users.dao import UsersDAO

router = APIRouter(prefix='/main', tags=['Главная страница'])
templates = Jinja2Templates(directory='app/templates')




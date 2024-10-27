from jose import jwt
from fastapi import APIRouter, HTTPException, status, Response, Depends, UploadFile, Request, File
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse


from users.schemas import SUserRegister, STokens

from users.dao import UsersDAO
from users.auth import get_password_hash
from users.auth import authenticate_user, create_access_token
from users.schemas import SUserAuth
from users.models import User
from users.dependencies import get_current_user
from databases.schemas import NewDB
from config import get_auth_data

from databases.dao import DatabasesDAO

router = APIRouter(prefix='/user', tags=['Работа с пользователями'])
templates = Jinja2Templates(directory='templates')

@router.get("/register")
async def register_user_template(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@router.post("/register")
async def register_user(user_data: SUserRegister, response: Response):
    user = await UsersDAO.find_one_or_none(email=user_data.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Пользователь с таким email уже существует')
    user = await UsersDAO.find_one_or_none(phone_number=user_data.phone_number)
    if user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Пользователь с таким номером телефона уже существует')
    user_dict = user_data.dict()
    user_dict['password'] = await get_password_hash(user_data.password)
    await UsersDAO.add(**user_dict)
    user = await UsersDAO.find_one_or_none(email=user_data.email)
    access_token = await create_access_token({"sub": str(user.id)})
    await UsersDAO.update(filter_by={'id': user.id}, token=access_token)
    response.set_cookie(key="users_access_token", value=access_token, httponly=True)
    return {"user_id": user.id, 'message':"ok"}


@router.get('/login')
async def get_students_html(request: Request):
    return templates.TemplateResponse(name='login.html', context={'request': request})

@router.post("/login")
async def auth_user(response: Response, user_data: SUserAuth):
    check = await authenticate_user(email=user_data.email, password=user_data.password)
    if check is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='Неверная почта или пароль')
    access_token = await create_access_token({"sub": str(check.id)})
    response.set_cookie(key="users_access_token", value=access_token, httponly=True)
    return {'access_token': access_token, 'refresh_token': None, 'message':"ok"}

@router.get("/profile")
async def get_me(request: Request, user_data: User = Depends(get_current_user)):
    return templates.TemplateResponse(name='profile.html', context={'request': request, 'profile':user_data})


@router.get("/buy_tokens")
async def buy_tokens_page(request: Request):
    return templates.TemplateResponse(name='buy_tokens.html', context={'request': request})

@router.put('/buy_tokens')
async def buy_tokens(count: STokens, user_data: User = Depends(get_current_user)) -> dict:
    check = await UsersDAO.update(filter_by={'id': user_data.id},
                                   tokens_count=user_data.tokens_count+count.tokens)
    if check:
        return {"message": "Токены успешно добавлены!"}
    else:
        return {"message": "Ошибка при добавлении токенов"}
@router.post("/new_project")
async def db_connect(db_info: NewDB, user_data: User = Depends(get_current_user)):
    db_dict = db_info.dict()
    db_dict['user_token'] = user_data.token
    db_dict['token'] = ''
    await DatabasesDAO.add(**db_dict)
    db = await DatabasesDAO.find_one_or_none(user_token=user_data.token)
    auth_data = get_auth_data()
    db_token = jwt.encode(db.id, auth_data['secret_key'], algorithm=auth_data['algorithm'])
    await DatabasesDAO.update(filter_by={'id': db.id},token=db_token)
    await UsersDAO.update(filter_by={'id': user_data.id}, databases=user_data.databases+[db.token])
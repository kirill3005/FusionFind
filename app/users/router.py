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

from users.auth import generate_single_part_token
from db_migration.migrate import DataMigration

router = APIRouter(prefix='/user')
templates = Jinja2Templates(directory='templates')


@router.post("/register", tags=['Регистрация нового пользователя (номер телефона вводить в правильном виде)'])
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

@router.post("/login", tags=['Авторизация пользователя'])
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
    return templates.TemplateResponse(name='profile.html', context={'request': request, 'profile':user_data, "databases": DatabasesDAO.find_all(user_token=user_data.token)})


@router.get("/buy_tokens")
async def buy_tokens_page(request: Request):
    return templates.TemplateResponse(name='buy_tokens.html', context={'request': request})

@router.put('/buy_tokens', tags=['Купить токены'])
async def buy_tokens(count: STokens, user_data: User = Depends(get_current_user)) -> dict:
    check = await UsersDAO.update(filter_by={'id': user_data.id},
                                   tokens_count=user_data.tokens_count+count.tokens)
    if check:
        return {"message": "Токены успешно добавлены!"}
    else:
        return {"message": "Ошибка при добавлении токенов"}

@router.get('/tokens_count',tags=['Запросить колво токенов'])
async def tokens_count(user_data: User = Depends(get_current_user)):
    return user_data.tokens_count

@router.get('/token', tags=['Запросить свой токен'])
async def get_token(request: Request, user_data: User = Depends(get_current_user)):
    return user_data.token

@router.get('/projects', tags=['Запросить свои проекты'])
async def get_projects(user_data: User = Depends(get_current_user)):
    return await DatabasesDAO.find_all(user_token=user_data.token)

@router.get('/new_project')
async def new_project_get(request: Request):
    return templates.TemplateResponse(name='new_project.html', context={'request': request})

@router.post("/new_project", tags=['Создать новый проект'])
async def db_connect(db_info: NewDB, user_data: User = Depends(get_current_user)):
    db_dict = db_info.dict()
    db_dict['user_token'] = user_data.token
    db_dict['token'] = ''
    await DatabasesDAO.add(**db_dict)
    db = (await DatabasesDAO.find_all(user_token=user_data.token))[-1]
    db_token = generate_single_part_token(db.id)
    await DatabasesDAO.update(filter_by={'id': db.id},token=db_token)
    await UsersDAO.update(filter_by={'id': user_data.id}, databases=user_data.databases+[db_token])
    config = {
        'database': {
            'dialect': db.dialect,  # Тип реляционной базы данных (может быть MySQL, SQLite, etc.)
            'host': db.host,  # Хост базы данных
            'port': int(db.port),  # Порт базы данных
            'user': db.user,  # Имя пользователя для базы данных
            'password': db.password,  # Пароль для подключения к базе данных
            'database': db.db_name  # Название базы данных
        },
        'qdrant': {
            'host': '87.249.44.115',  # Хост для подключения к Qdrant
            'port': 6333,  # Порт для подключения к Qdrant
            'collection_name': db_token,  # Имя коллекции в Qdrant
            'vector_size': 1024  # Размер векторов (должен соответствовать модели)
        },
        'mapping': {
            'table': db.table_name,  # Имя таблицы в реляционной базе данных
            'vector_column': db.vector_column,  # Колонка с текстом для векторизации
            'image_column': db.image_column,
            'metadata_columns': db.metadata_columns,  # Колонки с метаданными
        },
        'image_save_path': './images',  # Директория для сохранения изображений
        'lang': 'en'  # Язык
    }
    migrator = DataMigration(config)
    migrator.migrate()
    return {'message':"OK"}

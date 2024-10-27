from datetime import datetime, date
from typing import Optional, List
import re
from pydantic import BaseModel, Field, EmailStr, validator, ConfigDict, field_validator



class NewDB(BaseModel):
    dialect: str = Field(..., description='Тип вашей реляционной базы данных')
    host: str = Field(..., description='Хост сервера, на которой размещена база данных')
    port: str = Field(..., description='Порт')
    user: str = Field(..., description='Логин пользователя базы данных')
    password: str = Field(..., description='Пароль')
    db_name: str = Field(..., description='Название базы данных товаров')




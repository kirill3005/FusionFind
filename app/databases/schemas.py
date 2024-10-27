from datetime import datetime, date
from typing import Optional, List
import re
from pydantic import BaseModel, Field, EmailStr, validator, ConfigDict, field_validator



class NewDB(BaseModel):
    dialect: str = Field(...)
    host: str = Field(...)
    port: str = Field(...)
    user: str = Field(...)
    password: str = Field(...)
    db_name: str = Field(...)




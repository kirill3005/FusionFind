from datetime import datetime, date
from typing import Optional, List
import re
from pydantic import BaseModel, Field, EmailStr, validator, ConfigDict, field_validator



class NewMessage(BaseModel):
    api_token: str
    db_token: str
    conversation_id: str
    message: str
    photo: str = None

class Score(BaseModel):
    adekv: int
    mistakes: int
    useful: int
    username: str
    user_msg: str
    model_msg: str
    product_image: str




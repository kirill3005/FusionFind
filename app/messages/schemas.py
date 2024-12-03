from datetime import datetime, date
from typing import Optional, List
import re
from pydantic import BaseModel, Field, EmailStr, validator, ConfigDict, field_validator



class NewMessage(BaseModel):
    api_token: str
    db_token: str
    conversation_id: int
    message: str
    photo: str = None





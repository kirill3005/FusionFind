from datetime import datetime, date
from typing import Optional, List
import re
from pydantic import BaseModel, Field, EmailStr, validator, ConfigDict, field_validator



class NewMessage(BaseModel):
    message: str = Field(...)
    photo: Optional[str] = Field(None)




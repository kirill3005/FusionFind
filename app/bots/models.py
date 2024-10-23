from datetime import date
from database import Base, str_uniq, int_pk, str_null_true
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy import ForeignKey, text, Text
class Bot(Base):
    id: Mapped[int_pk]
    user_id: Mapped[int]

    extend_existing = True


from datetime import date
from database import Base, str_uniq, int_pk, str_null_true
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey, text, Text
class User(Base):
    id: Mapped[int_pk]
    password: Mapped[str_null_true]
    phone_number: Mapped[str_uniq]
    email: Mapped[str_uniq]
    first_name: Mapped[str]
    last_name: Mapped[str]
    token: Mapped[str_uniq]
    tokens_count: Mapped[int]


    extend_existing = True

    def __str__(self):
        return (f"{self.__class__.__name__}(id={self.id}, "
                f"first_name={self.first_name!r},"
                f"last_name={self.last_name!r})")
    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id})"
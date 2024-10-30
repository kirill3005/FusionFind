from datetime import date
from database import Base, str_uniq, int_pk, str_null_true
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy import ForeignKey, text, Text, Column, ARRAY, String
class Database(Base):
    id: Mapped[int_pk]
    token: Mapped[str]
    user_token: Mapped[str]
    dialect: Mapped[str]
    host: Mapped[str]
    port: Mapped[str]
    user: Mapped[str]
    password: Mapped[str]
    db_name: Mapped[str]
    table_name: Mapped[str]
    vector_column: Mapped[str]
    metadata_columns = Column(ARRAY(String), default=list)
    image_column: Mapped[str]

    extend_existing = True


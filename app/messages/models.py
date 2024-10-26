from datetime import date
from database import Base, str_uniq, int_pk, str_null_true
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy import ForeignKey, text, Text
class Message(Base):
    id: Mapped[int_pk]
    conversation_id: Mapped[int]
    user_token: Mapped[str]
    message: Mapped[str]
    photo: Mapped[str]
    sender: Mapped[str]

    extend_existing = True

class Conversation(Base):
    id: Mapped[int_pk]
    user_token: Mapped[str]
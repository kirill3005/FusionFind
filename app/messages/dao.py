from dao.base import BaseDAO
from messages.models import Message

from messages.models import Conversation


class MessagesDAO(BaseDAO):
    model = Message

class ConversationsDAO(BaseDAO):
    model = Conversation
from dao.base import BaseDAO
from messages.models import Message

from messages.models import Conversation

from messages.models import Scores


class MessagesDAO(BaseDAO):
    model = Message

class ConversationsDAO(BaseDAO):
    model = Conversation

class ScoresDAO(BaseDAO):
    model = Scores
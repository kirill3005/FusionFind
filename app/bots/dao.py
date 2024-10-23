from dao.base import BaseDAO
from bots.models import Bot


class BotDAO(BaseDAO):
    model = Bot
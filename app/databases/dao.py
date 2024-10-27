from dao.base import BaseDAO
from databases.models import Database




class DatabasesDAO(BaseDAO):
    model = Database


from fastapi import FastAPI, Request, Response

from app.users.dao import UsersDAO

app = FastAPI()


@app.get('/')
async def index():
    return {'message': 'Hello World'}

@app.post('/')
async def main_api(request: Request, user_token: str):
    user = UsersDAO.find_one_or_none(token=user_token)
    return user
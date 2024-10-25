from fastapi import FastAPI, Request, Response

from users.dao import UsersDAO

app = FastAPI()


@app.get('/')
async def index():
    return {'message': 'Hello World'}

@app.post('/')
async def main_api(user_token: str):
    user = await UsersDAO.find_one_or_none(token=user_token)
    return user
from fastapi import FastAPI, Request, Response, Header

from users.dao import UsersDAO

app = FastAPI()


@app.get('/')
async def index():
    return {'message': 'Hello World'}

@app.post('/{token}')
async def main_api(token: str):
    user = await UsersDAO.find_one_or_none(token=token)
    return user
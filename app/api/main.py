from fastapi import FastAPI, Request, Response, Header

from users.dao import UsersDAO

from messages.dao import MessagesDAO, ConversationsDAO

from messages.schemas import NewMessage

app = FastAPI()


@app.get('/')
async def index():
    return {'message': 'Hello World'}


@app.post('/{token}')
async def main_api(token: str):
    user = await UsersDAO.find_one_or_none(token=token)
    return user


@app.post('/new_conversation/{token}')
async def new_conv(token: str):
    await ConversationsDAO.add(**{'user_token': token})


@app.post('/message/{token}/{conversation_id}')
async def send_message(token: str, conversation_id: int, message: NewMessage):
    user = await UsersDAO.find_one_or_none(token=token)
    if user.tokens_count <= 0:
        return {'message': 'У вас закончились токены'}
    await UsersDAO.update(filter_by={'token': token},tokens_count=user.tokens_count - 1)
    msg_dict = message.dict()
    msg_dict['user_token'] = token
    msg_dict['sender'] = 'user'
    msg_dict['conversation_id'] = conversation_id
    await MessagesDAO.add(**msg_dict)
    '''response = model(message.message, message.photo)'''
    response_dict = {'message': 'response', 'user_token': token, 'photo': '', 'sender': 'model',
                     'conversation_id': conversation_id}
    await MessagesDAO.add(**response_dict)
    return 'response'

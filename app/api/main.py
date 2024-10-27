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


@app.post('/new_conversation/{token}/{project_token}',tags=['Создать новый диалог'])
async def new_conv(token: str, project_token: str):
    await ConversationsDAO.add(**{'user_token': token, 'project_token': project_token})


@app.post('/message/{token}/{project_token}/{conversation_id}',tags=['Передача сообщения от пользователя модели и получение ответа'])
async def send_message(token: str, project_token:str, conversation_id: int, message: NewMessage):
    user = await UsersDAO.find_one_or_none(token=token)
    if user.tokens_count <= 0:
        return {'message': 'У вас закончились токены'}
    await UsersDAO.update(filter_by={'token': token},tokens_count=user.tokens_count - 1)
    msg_dict = message.dict()
    msg_dict['user_token'] = token
    msg_dict['sender'] = 'user'
    msg_dict['conversation_id'] = conversation_id
    msg_dict['project_token'] = project_token
    await MessagesDAO.add(**msg_dict)
    '''response = model(message.message, message.photo)'''
    response_dict = {'message': 'response', 'user_token': token, 'photo': '', 'sender': 'model',
                     'conversation_id': conversation_id, 'project_token': project_token}
    await MessagesDAO.add(**response_dict)
    return 'Ответ от модели'
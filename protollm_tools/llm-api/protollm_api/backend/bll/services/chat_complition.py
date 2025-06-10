from protollm_api.backend.bll.services.base import BaseBLLService


class ChatCompletionService(BaseBLLService):
    def __init__(self):
        pass

    async def get_chat_completion(self, request) -> str:
        """Вся логика запроса chat_completion, тут же обрабатывает ответ и ошибки (иил бросаем новые)"""
        pass


import logging
from pydantic import BaseModel
import uuid

from typing import List, Tuple
from typing import Optional, Any, Annotated
from typing import AsyncGenerator

from langgraph.prebuilt import create_react_agent


from transformers import PreTrainedTokenizerFast
from langchain_core.tools import StructuredTool

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
    SystemMessage,
)

from protollm_agents.sdk.base import Event
from protollm_agents.sdk.events import TextEvent, ErrorEvent, MultiDictEvent


logger = logging.getLogger(__name__)

SYS_PROMPT = """ Ты виртуальный ассистент, который занимается перенаправлением запросов пользователей к соответствующим инструментам. \
Первым сообщением пользователю сообщи план обработки его запроса, какие инструменты ты будешь использовать и в каком порядке. Обязательно предложи запасной вариант - инструмент, которым ты воспользуешься, если первый не даст необходимый результат. \
Тебе запрещено называть инструменты по имени функций, но ты можешь давать их описание. Например, вместо save_document, скажи функция сохранения документа.\
Если необходимого инструмента нет в списке - попытайся ответить на вопрос самостоятельно, при этом сообщив пользователю о том, что у тебя нет соответствующего инструмента. \
Есть инструменты, которые используют сообщения чата для работы, поэтому тебе нужно писать в сообщениях только по одной теме и только релевантную информацию. Не пиши в обычных ответах пользователю приглашений к диалогу и обоснований выбора.
"""


def _take_any(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    return a or b


class MaxInputTokensExceededException(Exception):
    pass


class EnsembleOutputs(BaseModel):
    # general
    question: Annotated[str, _take_any]
    chat_history: Annotated[Optional[List[BaseMessage]], _take_any] = None

    # Tools results
    tool_results: Annotated[Optional[List[ToolMessage]], _take_any] = None

    # answer generating part
    all_answers: Annotated[Optional[List[AIMessage]], _take_any] = None
    all_messages: Annotated[Optional[List[BaseMessage]], _take_any] = None
    answer: Annotated[Optional[str], _take_any] = None

    # general if error occurs anywhere
    error: Annotated[Optional[Exception], _take_any] = None

    class Config:
        arbitrary_types_allowed = True


class EnsembleRouterPipeline:

    def __init__(
        self,
        *,
        agent_id: uuid.UUID,
        model: ChatOpenAI,
        tools: List[StructuredTool],
        tokenizer: PreTrainedTokenizerFast,
        max_input_tokens: int = 6144,
        max_chat_history_token_length: int = 24576,
    ):
        self._agent_id = agent_id
        self._tools = tools
        self._tokenizer = tokenizer
        self._max_input_tokens = max_input_tokens
        self._max_chat_history_token_length = max_chat_history_token_length

        self._last_message = None

        chat_model = model
        self._agent = create_react_agent(
            chat_model, self._tools, messages_modifier=SYS_PROMPT
        )
        self._config = {}

    async def invoke(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage] | List[Tuple[str, str]]] = None,
        raise_if_error: bool = False,
    ) -> EnsembleOutputs:
        chat_history = self._convert_history(chat_history)
        question_message = HumanMessage(content=question)
        chat_history = (chat_history or []) + [question_message]
        output = []
        async for event in self._agent.astream(
            {"messages": chat_history}, config=self._config
        ):
            output.append(list(event.values())[0]["messages"][0])
            self._last_message = output[-1].content

        tool_results = [msg for msg in output if isinstance(msg, ToolMessage)]
        answers = [msg for msg in output if isinstance(msg, AIMessage)]

        return EnsembleOutputs(
            question=question,
            chat_history=chat_history[:-1],
            tool_results=tool_results,
            all_answers=answers,
            all_messages=output,
            answer=answers[-1].content,
        )

    async def stream(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage] | List[Tuple[str, str]]] = None,
        raise_if_error: bool = False,
        only_eos_answer_event: bool = False,
    ) -> AsyncGenerator[Event, None]:
        self._check_num_input_tokens(question)
        current_output = ""
        tools_output = ""
        chat_history = self._convert_history(chat_history)
        chat_history = self._cut_history(
            chat_history, self._max_chat_history_token_length, self._tokenizer
        )
        question_message = HumanMessage(content=question)
        chat_history = (chat_history or []) + [question_message]
        try:
            async for event in self._agent.astream_events(
                {"messages": chat_history},
                config=self._config,
                debug=True,
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    delta = event["data"]["chunk"].content
                    if delta:
                        current_output += delta
                        stream_event = TextEvent(
                            agent_id=self._agent_id,
                            name="answer",
                            result=current_output,
                            is_eos=False,
                        )
                elif (
                    event["event"] == "on_chain_stream"
                    and "make_context" in event["name"]
                ):
                    retrieved_documents = event["data"]["chunk"].retrieved_documents
                    if not retrieved_documents:
                        stream_event = None
                    else:
                        stream_event = MultiDictEvent(
                            agent_id=self._agent_id,
                            name="retrieval",
                            result=[doc.dict() for doc in retrieved_documents],
                        )
                elif event["event"] == "on_llm_stream":
                    delta = event["data"]["chunk"].text
                    if delta:
                        tools_output += delta
                        stream_event = TextEvent(
                            agent_id=self._agent_id,
                            name="tool_answer",
                            result=tools_output,
                            is_eos=False,
                        )
                elif event["event"] == "on_chain_end":
                    if (
                        isinstance(event["data"]["output"], dict)
                    ) and "messages" in event["data"]["output"]:
                        self._last_message = event["data"]["output"]["messages"][
                            0
                        ].content
                    if event["name"] == "LangGraph":
                        stream_event = TextEvent(
                            agent_id=self._agent_id,
                            name="answer",
                            result=current_output,
                            is_eos=True,
                        )
                else:
                    stream_event = None
                if stream_event:
                    yield stream_event
                    if isinstance(stream_event, ErrorEvent):
                        if raise_if_error:
                            raise ValueError(stream_event.result)
                        else:
                            yield stream_event
        except Exception as ex:
            if raise_if_error:
                raise ex
            yield ErrorEvent(result=str(ex), is_eos=True)

    @staticmethod
    def _convert_history(
        chat_history: Optional[List[BaseMessage] | List[Tuple[str, str]]],
    ) -> List[BaseMessage]:
        def _convert_tuple(msg_type: str, msg: str) -> BaseMessage:
            supported_msg_types = ["system", "human", "ai"]

            match msg_type:
                case "system":
                    message = SystemMessage(content=msg)
                case "human":
                    message = HumanMessage(content=msg)
                case "ai":
                    message = AIMessage(content=msg)
                case _:
                    raise ValueError(
                        f"Unsupported message type for ({msg_type}, {msg}). "
                        f"Supported types: {supported_msg_types}"
                    )

            return message

        if (
            not chat_history
            or len(chat_history) == 0
            or isinstance(chat_history[0], BaseMessage)
        ):
            return chat_history
        elif isinstance(chat_history[0], tuple):
            return [_convert_tuple(msg_type, msg) for msg_type, msg in chat_history]
        else:
            raise ValueError(
                "Unsupported types of list values for chat_history. "
                "It should be either BaseMessage or Tuple"
            )

    @staticmethod
    def _cut_history(
        chat_history: Optional[List[BaseMessage] | List[Tuple[str, str]]],
        threshold: int,
        tokenizer: PreTrainedTokenizerFast,
    ) -> List[BaseMessage]:
        if not chat_history or len(chat_history) == 0:
            return chat_history
        res = []
        s = 0
        for message in reversed(chat_history):
            l = len(tokenizer.tokenize(message.content))
            if s + l > threshold:
                break
            s += l
            res.append(message)
        return res[::-1]

    def _check_num_input_tokens(self, question: str):
        tokens = self._tokenizer.tokenize(question)
        if len(tokens) > self._max_input_tokens:
            raise MaxInputTokensExceededException(
                f"Input length {len(tokens)} exceeds allowed max number of tokens {self._max_input_tokens}."
            )

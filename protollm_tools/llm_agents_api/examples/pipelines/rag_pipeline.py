from typing import AsyncGenerator


import ast
import datetime
import logging
from dataclasses import dataclass, fields
from typing import List, Tuple, Dict, cast, Callable, Awaitable
from typing import Optional, Any, Annotated, AsyncIterator, AsyncGenerator
from pydantic import BaseModel
import uuid

import numpy as np
from langchain.output_parsers.retry import RetryOutputParser
from langchain_community.llms.vllm import VLLMOpenAI
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableParallel, RunnableLambda, Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.utils import Input, Output
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.pregel.io import AddableValuesDict
from transformers import PreTrainedTokenizerFast

from protollm_agents.sdk.base import Event
from protollm_agents.sdk.events import TextEvent, ErrorEvent, MultiDictEvent


from langfuse.callback import CallbackHandler
from examples.pipelines.utils import RunnablePiece


logger = logging.getLogger(__name__)

PIPE_PREFIX = "rag_"
PIPE_CONTEXTUALIZER = f"{PIPE_PREFIX}question_contextualizer"
PIPE_PLANNER = f"{PIPE_PREFIX}planner_logic_chain"
PIPE_RETRIEVER = f"{PIPE_PREFIX}retriever_vector"
PIPE_MAKE_CONTEXT = f"{PIPE_PREFIX}make_context"
PIPE_ANSWER_GENERATOR = f"{PIPE_PREFIX}answer_generator"
PIPE_WORKFLOW = f"{PIPE_PREFIX}workflow"

default_prompts = {
    "planner_prompt_template": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """"
Ты система осуществляющая планирование запросов нужной для пользователя информации в векторной базе с документами разделенными на небольшие сегменты. \
Для каждого поступившего запроса от пользователя \
твоя задача, при необходимости, разделить его на несколько запросов к векторной базе данных для получения нужной для ответа информации. \
Ты должен написать итоговые запросы "Запросы:", а потом написать объяснение "Объяснение:". И запросы и объяснение должны быть на русском языке. \
Например, вопрос "назови столицу России" не нуждается в разделении, необходимая информация о столице России будет извлечена. \
В свою очередь, запрос "Сравни автомобили ауди q5 и audi q7" уже должен быть разделен на два: "Какие преимущества и недостатки и audi q5" и "Какие преимущества и недостатки у audi q7", т.к. нам \
нужна информация по обоим объектам для проведения сравнения. 
Пользуйся следующим набором правил:
1. Если будешь отвечать на английском получишь штраф.
2. Учитывай что у пользовательского запроса есть контекст, который нельзя терять.
3. Если пользовательский запрос можно разделить на несколько, не забудь добавить необходимый контекст для независимого поиска по базе.

Результат выведи списком запросов после ключевого слова ЗАПРОСЫ: ["запрос1", "запрос2"...]""",
            ),
            ("user", "{question}"),
            ("ai", ""),
        ]
    ),
    "chat_answer_prompt_template": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты - система отвечающая на вопросы пользователей по источникам. "
                "Тебе представлен вопрос пользователя и набор источников. "
                "Для каждого источника может быть указана дата его публикации. "
                "Используй ее для указания в ответе точного года, "
                "если в источнике есть указания на прошлые или будущие годы."
                "Для ответа используй только информацию в представленных источниках. "
                "Если на вопрос нельзя ответить исходя из источников "
                'напиши "не достаточно информации для ответа".',
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{context}\nВопрос: {question}\n"),
            ("ai", "Ответ:"),
        ]
    ),
    "chat_no_context_answer_prompt_template": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты полезный помощник, ведущий разговор с пользователем. Отвечай только на русском, иначе получишь штраф.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{context}"),
            ("ai", "Ответ:"),
        ]
    ),
    "contextualize_q_prompt": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Учитывая историю чата и последний вопрос пользователя, "
                "который может ссылаться на контекст в истории чата,"
                "переформулируй этот вопрос так, чтобы его можно было понять без истории чата. "
                "НЕ отвечай на вопрос, просто переформулируй его, если необходимо, "
                "а в противном случае верни его как есть.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
            ("ai", ""),
        ]
    ),
}


def _take_any(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    return a or b


def qwen2_convert(x: PromptValue) -> StringPromptValue:
    def _convert(msg: BaseMessage, last_message: bool) -> str:
        match msg:
            case SystemMessage():
                return f"<|im_start|>system\n{msg.content}\n<|im_end|>"
            case HumanMessage():
                return f"<|im_start|>user\n{msg.content}\n<|im_end|>"
            case AIMessage():
                if not last_message:
                    return f"<|start_header_id|>assistant<|end_header_id|>\n{msg.content}\n<|im_end|>"
                else:
                    return f"<|im_start|>assistant\n{msg.content}"
            case _:
                raise ValueError(f"Unsupported message type: {type(msg)}")

    messages = x.to_messages()

    text = "\n".join(
        [_convert(msg, i == len(messages) - 1) for i, msg in enumerate(messages)]
    )
    return StringPromptValue(text=text)


def _doc2str(doc: Document) -> str:
    return (
        f"Следующий фрагмент относится к файлу {doc.metadata.get('filename', '')} "
        f"и главе {doc.metadata.get('chapter', '')}."
        f"\n{doc.page_content}"
    )


async def aqwen2_convert(x: PromptValue) -> StringPromptValue:
    return qwen2_convert(x)


class ChainOutput(BaseModel):
    # general
    question: Annotated[str, _take_any]
    chat_history: Annotated[Optional[List[BaseMessage]], _take_any] = None

    # prepration: question contextualization, context expansion
    contextualized_question: Annotated[Optional[str], _take_any] = None

    # planner part (only for vector store)
    planned_queries: Annotated[Optional[List[str]], _take_any] = None

    # vector store part
    retrieved_documents: Annotated[Optional[List[Document]], _take_any] = None

    # answer generating part
    answer_context: Annotated[Optional[str], _take_any] = None
    answer: Annotated[Optional[str], _take_any] = None

    # general if error occurs anywhere
    error: Annotated[Optional[Exception], _take_any] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def contexts(self) -> Optional[List[Document]]:
        return self.retrieved_documents


@dataclass
class RAGPrompts:
    contextualize_q_prompt: Optional[ChatPromptTemplate] = None
    planner_prompt_template: Optional[ChatPromptTemplate] = None
    chat_answer_prompt_template: Optional[ChatPromptTemplate] = None
    chat_no_context_answer_prompt_template: Optional[ChatPromptTemplate] = None


@dataclass
class _RAGChains:
    contextualize_q_chain: Optional[Runnable[Dict[str, str], str]] = None
    planner_chain: Optional[Runnable[Dict[str, str], List[str]]] = None
    chroma_retriever_chain: Optional[Runnable[List[str], List[Document]]] = None
    answer_generator_chain: Optional[Runnable[Dict[str, str], str]] = None

    def call_with(
        self,
        func: Callable[
            ["_RAGChains", ChainOutput, RunnableConfig], Awaitable[ChainOutput]
        ],
    ) -> Callable[[ChainOutput], Awaitable[ChainOutput]]:
        async def _func(state: ChainOutput, config: RunnableConfig):
            return await func(self, state, config)

        return cast(Callable[[ChainOutput], Awaitable[ChainOutput]], _func)


class PlannerOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        """Parse by splitting."""
        try:
            result = ast.literal_eval(
                text.split("ЗАПРОСЫ:")[1].strip().split("]")[0] + "]"
            )
        except Exception as ex:
            logger.warning("Planner output parsing exception!", exc_info=True)
            raise OutputParserException(str(ex))
        return result


class MultiQueryRetriever:
    def __init__(self, retriever: List[VectorStoreRetriever] | VectorStoreRetriever):
        self._retrievers = (
            [retriever] if isinstance(retriever, VectorStoreRetriever) else retriever
        )

    @staticmethod
    def remove_duplicates(docs: List[Document]) -> List[Document]:
        return list({doc.page_content: doc for doc in docs}.values())

    def do_batch(self, queries: List[str]) -> List[Document]:
        result = [
            doc
            for retriever in self._retrievers
            for docs in retriever.batch(queries)
            for doc in docs
        ]
        result = self.remove_duplicates(result)
        return result

    async def do_abatch(self, arr: List[str]) -> List[Document]:
        result = [
            doc
            for retriever in self._retrievers
            for docs in (await retriever.abatch(arr))
            for doc in docs
        ]

        result = self.remove_duplicates(result)
        return result


class MaxInputTokensExceededException(Exception):
    pass


class RAGPipeline:
    def __init__(
        self,
        *,
        agent_id: uuid.UUID,
        planner_model: Runnable,
        generator_model: Runnable,
        tokenizer: PreTrainedTokenizerFast,
        store: Optional[List[VectorStore] | VectorStore] = None,
        max_input_tokens: int = 6144,
        max_chat_history_token_length: int = 24576,
        prompts: Optional[RAGPrompts] = None,
        retrieving_top_k: int = 3,
        generator_context_top_k: Optional[int] = None,
        runnable_config: Optional[RunnableConfig] = None,
        include_original_question_in_queries: bool = True,
        langfuse_handler: Optional[CallbackHandler] = None,
        plan_queries: bool = False,
    ):
        self.agent_id = agent_id
        self.plan_queries = plan_queries
        self._planner_model = planner_model
        self._generator_model = generator_model
        self._tokenizer = tokenizer
        self._stores = [store] if isinstance(store, VectorStore) else store
        self._max_input_tokens = max_input_tokens
        self._max_chat_history_token_length = max_chat_history_token_length
        self._prompts = prompts
        if not self._prompts:
            self._prompts = RAGPrompts()
        for field in fields(self._prompts):
            attr_name = field.name
            if (
                attr_name in default_prompts
                and getattr(self._prompts, attr_name) is None
            ):
                print(f"setting {attr_name}")
                setattr(self._prompts, attr_name, default_prompts[attr_name])
        if any(
            getattr(self._prompts, field.name) is None
            for field in fields(self._prompts)
        ):
            for field in fields(self._prompts):
                print(field.name, getattr(self._prompts, field.name))
            raise ValueError("Some fields in the Prompts dataclass are still None!")

        self._retrieving_top_k = retrieving_top_k
        self._generator_context_top_k = (
            generator_context_top_k
            if generator_context_top_k is not None
            else retrieving_top_k
        )

        self._runnable_config = runnable_config
        self._retrieving_top_k = retrieving_top_k
        self._final_state: Optional[ChainOutput] = None
        self._include_original_question_in_queries = (
            include_original_question_in_queries
        )
        self._langfuse_handler = langfuse_handler

    def _get_planner_llm(self) -> BaseLLM:
        return cast(BaseLLM, self._planner_model)

    def _get_answer_generator_llm(self) -> BaseLLM:
        return cast(BaseLLM, self._generator_model)

    def _get_llm_prompt_converter(self) -> Runnable[PromptValue, StringPromptValue]:
        return RunnableLambda(qwen2_convert, afunc=aqwen2_convert)

    def _get_contextualization_chain(self) -> Runnable[Dict[str, str], str]:
        chain = (
            self._prompts.contextualize_q_prompt
            | self._get_llm_prompt_converter()
            | self._get_planner_llm()
        )
        return chain

    def _get_planner_chain(self) -> Runnable[Dict[str, str], List[str]]:
        planner_llm = self._get_planner_llm()
        retry_planner_parser = RetryOutputParser.from_llm(
            parser=PlannerOutputParser(),
            llm=planner_llm,
            prompt=PromptTemplate.from_template("{prompt}"),
        )

        def _do_planning(x: dict):
            llm_out = x["llm_out"]
            pipe_input = x["pipe_input"]

            try:
                questions = retry_planner_parser.parse_with_prompt(**llm_out)
            except OutputParserException:
                prompt_value = cast(PromptValue, llm_out["prompt_value"])
                questions = [pipe_input["question"]]
                logger.warning(
                    "Procceeding without extended set of questions due to OutputParserException: "
                    "questions - %s, prompt - %s" % (questions, prompt_value)
                )

            return questions

        chain = (
            self._prompts.planner_prompt_template
            | self._get_llm_prompt_converter()
            | RunnableParallel(
                completion=planner_llm, prompt_value=RunnablePassthrough()
            )
        )
        planner_chain = RunnablePiece(
            step=(
                RunnableParallel(llm_out=chain, pipe_input=RunnablePassthrough())
                | RunnableLambda(_do_planning, name="retry_planner_lambda")
            ),
            name=PIPE_PLANNER,
        )

        return planner_chain

    def _get_retriever_chain(self) -> Optional[Runnable[List[str], List[Document]]]:
        if self._stores is None:
            return None

        search_kwargs = {"k": self._retrieving_top_k}

        mq_retriever = MultiQueryRetriever(
            retriever=[
                store.as_retriever(search_kwargs=search_kwargs)
                for store in self._stores
            ]
        )

        multiquery_retriever = RunnableLambda(
            func=mq_retriever.do_batch,
            afunc=mq_retriever.do_abatch,
            name=PIPE_RETRIEVER,
        )

        return multiquery_retriever

        def _do_planning(x: dict):
            llm_out = x["llm_out"]
            pipe_input = x["pipe_input"]

            try:
                questions = retry_planner_parser.parse_with_prompt(**llm_out)
            except OutputParserException:
                prompt_value = cast(PromptValue, llm_out["prompt_value"])
                questions = [pipe_input["question"]]
                logger.warning(
                    "Procceeding without extended set of questions due to OutputParserException: "
                    "questions - %s, prompt - %s" % (questions, prompt_value)
                )

            return questions

    def _get_answer_chain(self) -> Runnable[Dict[str, str], str]:
        answer_generator_chain = (
            self._prompts.chat_answer_prompt_template.bind(
                year=datetime.datetime.now().year
            )
            | self._get_llm_prompt_converter()
            | self._get_answer_generator_llm()
        )
        return answer_generator_chain

    async def _acontextualize_question(
        self, chains: _RAGChains, state: ChainOutput, config: RunnableConfig
    ) -> ChainOutput:
        if state.chat_history:
            contextualized_question = await chains.contextualize_q_chain.ainvoke(
                input={"question": state.question, "chat_history": state.chat_history},
                config=config,
            )
        else:
            contextualized_question = state.question

        return state.copy(update={"contextualized_question": contextualized_question})

    async def _aplanning(
        self, chains: _RAGChains, state: ChainOutput, config: RunnableConfig
    ) -> ChainOutput:
        if chains.chroma_retriever_chain is not None:
            queries = await chains.planner_chain.ainvoke(
                {
                    "question": state.contextualized_question,
                },
                config=config,
            )
            queries = (
                [state.contextualized_question, *queries]
                if self._include_original_question_in_queries
                else queries
            )
        else:
            queries = [state.contextualized_question]

        result = state.copy(
            update={
                "planned_queries": queries,
            }
        )
        return result

    async def _aretrieving(
        self, chains: _RAGChains, state: ChainOutput, config: RunnableConfig
    ) -> ChainOutput:
        if chains.chroma_retriever_chain is None:
            return state
        docs = await chains.chroma_retriever_chain.ainvoke(
            state.planned_queries, config=config
        )
        return state.copy(update={"retrieved_documents": docs})

    async def _amake_context(
        self, chains: _RAGChains, state: ChainOutput, config: RunnableConfig
    ) -> ChainOutput:
        paragraphs = [
            f"Источник {i + 1}: {_doc2str(doc)}"
            for i, doc in enumerate(state.retrieved_documents)
        ]

        if self._generator_context_top_k > 0:
            paragraphs = paragraphs[: self._generator_context_top_k]

        logger.debug("Available paragraphs for context building: %s" % len(paragraphs))

        if len(paragraphs) > 0:
            psizes = np.cumsum([len(self._tokenizer.tokenize(p)) for p in paragraphs])
            if psizes[-1] > self._max_input_tokens:
                idx = np.argmax(psizes > self._max_input_tokens)
                logger.warning(
                    "Reducing number of paragraphs in the context due to limit on tokens: "
                    f"all paragraphs size %s > %s "
                    f"Selecting first chunks %s / %s."
                    % (psizes[-1], self._max_input_tokens, idx + 1, len(paragraphs))
                )
            else:
                idx = len(psizes) - 1
            paragraphs = paragraphs[: idx + 1]

        logger.info(
            "Num paragraphs fitting to the network context: %s" % len(paragraphs)
        )

        paragraphs_contexts = "\n".join(paragraphs)

        contexts = "\n\n".join([el for el in paragraphs_contexts if el])
        return state.copy(update={"answer_context": contexts})

    async def _agenerate_answer(
        self, chains: _RAGChains, state: ChainOutput, config: RunnableConfig
    ) -> ChainOutput:
        answer = await chains.answer_generator_chain.ainvoke(
            input={
                "question": state.contextualized_question,
                "context": state.answer_context,
                "chat_history": [],
            },
            config=config,
        )
        return state.copy(update={"answer": answer})

    @staticmethod
    def _convert_pipeline_result(
        question: str, result: AddableValuesDict
    ) -> ChainOutput:
        return (
            ChainOutput(question=question, error=result)
            if isinstance(result, Exception)
            else ChainOutput(**result)
        )

    def _build(self) -> Runnable[ChainOutput, AddableValuesDict]:
        workflow = StateGraph(ChainOutput)

        cs = _RAGChains(
            contextualize_q_chain=self._get_contextualization_chain(),
            planner_chain=self._get_planner_chain(),
            chroma_retriever_chain=self._get_retriever_chain(),
            answer_generator_chain=self._get_answer_chain(),
        )

        workflow.add_node(
            PIPE_CONTEXTUALIZER, cs.call_with(self._acontextualize_question)
        )
        workflow.add_node(PIPE_PLANNER, cs.call_with(self._aplanning))
        workflow.add_node(PIPE_RETRIEVER, cs.call_with(self._aretrieving))
        workflow.add_node(PIPE_MAKE_CONTEXT, cs.call_with(self._amake_context))
        workflow.add_node(PIPE_ANSWER_GENERATOR, cs.call_with(self._agenerate_answer))

        workflow.add_edge(START, PIPE_CONTEXTUALIZER)
        workflow.add_edge(PIPE_CONTEXTUALIZER, PIPE_PLANNER)
        workflow.add_edge(PIPE_PLANNER, PIPE_RETRIEVER)

        workflow.add_edge(PIPE_RETRIEVER, PIPE_MAKE_CONTEXT)

        workflow.add_edge(PIPE_MAKE_CONTEXT, PIPE_ANSWER_GENERATOR)
        workflow.add_edge(PIPE_ANSWER_GENERATOR, END)

        wf = workflow.compile()
        wf.name = PIPE_WORKFLOW
        return wf

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

    async def stream(
        self,
        question: str,
        chat_history: list[tuple[str, str]],
        raise_if_error: bool = False,
    ) -> AsyncGenerator[Event, None]:
        current_output = ""
        chain = self._build()
        try:
            self._check_num_input_tokens(question)
            chat_history = self._convert_history(chat_history)
            chat_history = self._cut_history(
                chat_history, self._max_chat_history_token_length, self._tokenizer
            )

            conf = (
                self._runnable_config
                if self._runnable_config
                else (
                    {"callbacks": [self._langfuse_handler]}
                    if self._langfuse_handler
                    else None
                )
            )
            async for event in chain.astream_events(
                ChainOutput(question=question, chat_history=chat_history),
                config=conf,
                version="v2",
                stream_mode="values",
            ):
                if (
                    event["event"] == "on_chain_stream"
                    and event["name"] == PIPE_MAKE_CONTEXT
                ):
                    state = cast(ChainOutput, event["data"]["chunk"])

                    stream_event = MultiDictEvent(
                        agent_id=self.agent_id,
                        name="retrieval",
                        result=[doc.dict() for doc in state.retrieved_documents],
                    )
                elif (
                    event["event"] == "on_llm_stream"
                    and event["name"] == self._generator_model.bound.name
                ):
                    delta = event["data"]["chunk"].text
                    current_output += delta
                    stream_event = TextEvent(
                        agent_id=self.agent_id,
                        name="answer",
                        result=current_output,
                        is_eos=False,
                    )
                elif (
                    event["event"] == "on_llm_end"
                    and event["name"] == self._generator_model.bound.name
                ):
                    stream_event = TextEvent(
                        agent_id=self.agent_id,
                        name="answer",
                        result=event["data"]["output"]["generations"][0][0]["text"],
                        is_eos=True,
                    )
                elif (
                    event["event"] == "on_chain_end" and event["name"] == PIPE_WORKFLOW
                ):
                    self._final_state = ChainOutput.parse_obj(event["data"]["output"])
                    stream_event = None
                    logger.debug(msg=str(self._final_state))
                else:
                    stream_event = None

                if stream_event:
                    if not isinstance(stream_event, TextEvent) or stream_event.is_eos:
                        logger.debug(f"Yielding stream event: {stream_event}")

                    yield stream_event
        except Exception as ex:
            if raise_if_error:
                raise ex
            yield ErrorEvent(agent_id=self.agent_id, result=str(ex))

    async def invoke(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage] | List[Tuple[str, str]]] = None,
        raise_if_error: bool = False,
    ) -> ChainOutput:
        chain = self._build()

        self._check_num_input_tokens(question)
        chat_history = self._convert_history(chat_history)
        chat_history = self._cut_history(
            chat_history, self._max_chat_history_token_length, self._tokenizer
        )

        conf = (
            self._runnable_config
            if self._runnable_config
            else (
                {"callbacks": [self._langfuse_handler]}
                if self._langfuse_handler
                else None
            )
        )
        result = await chain.ainvoke(
            ChainOutput(question=question, chat_history=chat_history),
            config=conf,
        )
        return self._convert_pipeline_result(question, result)

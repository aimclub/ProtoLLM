import asyncio
import threading
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Any,
    Type,
    Dict,
    Iterator,
    AsyncIterator,
    Sequence,
    Generic,
    TypeVar,
)
from uuid import UUID

from langchain_core.callbacks import (
    CallbackManagerForChainRun,
    AsyncCallbackManagerForChainRun,
    BaseCallbackHandler,
)
from langchain_core.documents import Document
from langchain_core.load import dumpd
from langchain_core.outputs import LLMResult
from langchain_core.runnables import (
    RunnableSerializable,
    Runnable,
    RunnableConfig,
    ensure_config,
)
from langchain_core.runnables.config import (
    get_executor_for_config,
    patch_config,
    get_async_callback_manager_for_config,
)
from langchain_core.runnables.utils import (
    Input,
    Output,
    AddableDict,
    indent_lines_after_first,
    ConfigurableFieldSpec,
    get_unique_config_specs,
)
from langchain_core.utils.aiter import atee
from langchain_core.utils.iter import safetee
from pydantic import BaseModel
from tqdm import tqdm


class RunnablePiece(RunnableSerializable[Input, Output]):
    step: Runnable[Input, Output]

    def __init__(
        self, step: Runnable[Input, Output], name: Optional[str] = None
    ) -> None:
        super().__init__(step=step, name=name)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    class Config:
        arbitrary_types_allowed = True

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        name = name or self.name or f"RunnableLogicPiece<{self.name}>"
        return super().get_name(suffix, name=name)

    @property
    def InputType(self) -> Any:
        if self.step.InputType:
            return self.step.InputType

        return Any

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        if all(
            s.get_input_schema(config).schema().get("type", "object") == "object"
            for s in self.steps__.values()
        ):
            # This is correct, but pydantic typings/mypy don't think so.
            return create_model(  # type: ignore[call-overload]
                self.get_name("Input"),
                **{
                    k: (v.annotation, v.default)
                    for step in self.steps__.values()
                    for k, v in step.get_input_schema(config).__fields__.items()
                    if k != "__root__"
                },
            )

        return super().get_input_schema(config)

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        # This is correct, but pydantic typings/mypy don't think so.
        return create_model(  # type: ignore[call-overload]
            self.get_name("Output"),
            **{k: (v.OutputType, None) for k, v in self.steps__.items()},
        )

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return get_unique_config_specs(spec for spec in self.step.config_specs)

    def __repr__(self) -> str:
        map_for_repr = (
            f"step: {indent_lines_after_first(repr(self.step), '  ' + 'step' + ': ')}"
        )
        return "{\n  " + map_for_repr + "\n}"

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        from langchain_core.callbacks.manager import CallbackManager

        # setup callbacks
        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # gather results from all steps
        try:
            # copy to avoid issues from the caller mutating the steps during invoke()
            with get_executor_for_config(config) as executor:
                future = executor.submit(
                    self.step.invoke,
                    input,
                    # mark each step as a child run
                    patch_config(
                        config,
                        callbacks=run_manager.get_child(f"map:key:step"),
                    ),
                )

                output = future.result()
        # finish the root run
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(output)
            return output

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Dict[str, Any]:
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # gather results from all steps
        try:
            # copy to avoid issues from the caller mutating the steps during invoke()
            output = await self.step.ainvoke(
                input,
                # mark each step as a child run
                patch_config(config, callbacks=run_manager.get_child(f"map:key:step")),
            )
        # finish the root run
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(output)
            return output

    def _transform(
        self,
        input: Iterator[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Iterator[AddableDict]:
        # Each step gets a copy of the input iterator,
        # which is consumed in parallel in a separate thread.
        input_copies = list(safetee(input, 1, lock=threading.Lock()))
        with get_executor_for_config(config) as executor:
            generator = self.step.transform(
                input_copies.pop(),
                patch_config(config, callbacks=run_manager.get_child(f"map:key:step")),
            )

            while True:
                future = executor.submit(next, generator)
                try:
                    yield future.result()
                except StopIteration:
                    break

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Dict[str, Any]]:
        yield from self.transform(iter([input]), config)

    async def _atransform(
        self,
        input: AsyncIterator[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> AsyncIterator[AddableDict]:

        # Each step gets a copy of the input iterator,
        # which is consumed in parallel in a separate thread.
        input_copies = list(atee(input, 1, lock=asyncio.Lock()))
        # Create the transform() generator for each step
        generator = self.step.atransform(
            input_copies.pop(),
            patch_config(config, callbacks=run_manager.get_child(f"map:key:step")),
        )

        async for chunk in generator:
            yield chunk

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_aiter(), config):
            yield chunk


InputsType = TypeVar("InputsType")
OutputsType = TypeVar("OutputsType")


@dataclass
class RunnableInfo:
    type: str
    parent_run_id: UUID
    name: Optional[str] = None
    inputs: Optional[Any] = None
    outputs: Optional[Any] = None
    error: Optional[Exception] = None

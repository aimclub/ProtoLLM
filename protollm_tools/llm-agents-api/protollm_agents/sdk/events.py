import enum

from pydantic import Field

from .base import Event


class EventType(str, enum.Enum):
    text = "text"
    multitext = "multitext"
    dict = "dict"
    multidict = "multidict"
    status = "status"
    error = "error"

class TextEvent(Event):
    name: str = EventType.text
    result: str = Field(..., description='Text of the event')

class MultiTextEvent(Event):
    name: str = EventType.multitext
    result: list[str] = Field(..., description='Sequence of texts of the event')

class DictEvent(Event):
    name: str = EventType.dict
    result: dict = Field(..., description='Arbitrary event')

class MultiDictEvent(Event):
    name: str = EventType.multidict
    result: list[dict] = Field(..., description='Sequence of dictionaries of the event')

class StatusEvent(Event):
    name: str = EventType.status
    result: str = Field(..., description='Status of the event')

class ErrorEvent(Event):
    name: str = EventType.error
    result: str = Field(..., description='Error of the event')


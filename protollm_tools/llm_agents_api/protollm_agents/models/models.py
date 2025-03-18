import datetime
import uuid

from sqlalchemy import TIMESTAMP, Column, String, types
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func
from typing_extensions import Annotated

str30 = Annotated[str, 30]
str50 = Annotated[str, 50]


class Base(DeclarativeBase):
    type_annotation_map = {
        datetime.datetime: TIMESTAMP(timezone=True),
        str30: String(length=30),
        str50: String(length=50),
    }


class Agent(Base):
    __tablename__ = "agents"

    agent_id: Mapped[uuid.UUID] = mapped_column(types.Uuid, primary_key=True)
    name: Mapped[str]
    description: Mapped[str]
    arguments = Column(JSONB, nullable=True)
    created: Mapped[datetime.datetime] = mapped_column(
        server_default=func.CURRENT_TIMESTAMP(), nullable=False
    )
    module_name: Mapped[str]
    class_name: Mapped[str]
    agent_type = Column(ARRAY(String), nullable=True)

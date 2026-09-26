from sqlalchemy import (
    String,
    Boolean,
    Integer,
    LargeBinary,
    ForeignKey
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from app.database.database import Base

class Member(Base):
    __tablename__ = "members"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )
    code: Mapped[str] = mapped_column(
        String(20),
        unique=True,
        nullable=False
    )
    full_name: Mapped[str] = mapped_column(
        String(70),
        nullable=False
    )
    room_number: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True
    )
    active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )
    member_id: Mapped[int] = mapped_column(
        ForeignKey("members.id"),
        nullable=False
    )
    pose: Mapped[str] = mapped_column(
        String(20),
        nullable=False
    )
    embedding: Mapped[bytes] = mapped_column(
        nullable=False    
    )
    embedding_dim: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    ) #lưu số phần tử
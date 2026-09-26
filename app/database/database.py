from sqlalchemy import (
    create_engine,
    URL
)

from sqlalchemy.orm import (
    DeclarativeBase,
    sessionmaker,
)

from app.config import (
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASSWORD,
    DB_NAME
)

database_url = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME
)

engine = create_engine(
    database_url,
    echo = True
)

class Base(DeclarativeBase):
    pass

SessionLocal = sessionmaker(
    bind= engine
)

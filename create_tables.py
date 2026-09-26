from app.database.database import (
    Base,
    engine
)

from app.database.models import (
    FaceEmbedding,
    Member
)

Base.metadata.create_all(
    bind=engine
)
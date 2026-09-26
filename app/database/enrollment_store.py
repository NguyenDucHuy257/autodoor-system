import numpy as np

from sqlalchemy import select

from app.database.database import SessionLocal
from app.database.models import (
    Member,
    FaceEmbedding,
)

def save_enrollment(
    member_info: dict,
    embeddings: dict[str, np.ndarray],
) -> int:

    session = SessionLocal()

    try:

        existing_member = session.scalar(
            select(Member).where(
                Member.code == member_info["code"]
            )
        )

        if existing_member is not None:
            raise ValueError(
                f"Mã thành viên {member_info['code']} đã tồn tại"
            )

        member = Member(
            code=member_info["code"],
            full_name=member_info["full_name"],
            room_number=member_info["room_number"],
        )

        session.add(member)

        session.flush()

        member_id = member.id

        for pose, embedding in embeddings.items():

            embedding = (
                embedding
                .reshape(-1)
                .astype(np.float32)
            )

            row = FaceEmbedding(
                member_id=member_id,
                pose=pose,
                embedding=embedding.tobytes(),
                embedding_dim=embedding.size,
            )

            session.add(row)

        session.commit()

        return member_id

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()
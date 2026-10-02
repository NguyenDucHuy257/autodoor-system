from sqlalchemy.orm import Session
from sqlalchemy import select, text

import numpy as np

from app.database.models import (
    Member,
    FaceEmbedding
)


class EnrollmentRepository:
    def __init__(
        self,
        session: Session
    ):
        self.session = session

    def find_member_by_code(
        self,
        code: str,
    ) -> Member | None:
        
        return self.session.scalar(
            select(Member).where(
                Member.code == code
            )
        )

    def create_member(
        self,
        code: str,
        full_name: str,
        room_number: str | None,

    ) -> Member:
        
        member = Member(
            code=code,
            full_name=full_name,
            room_number=room_number
        )

        self.session.add(member)

        self.session.flush()

        return member

    def create_face_embedding(
        self,
        member_id: int,
        pose: str,
        embedding: np.ndarray,
    ) -> FaceEmbedding:

        embedding = (
            embedding
            .reshape(-1)
            .astype(np.float32)
        )

        if embedding.size == 0 and np.isfinite(embedding):
            raise ValueError(f"Embedding của member id {member_id} rỗng!")

        row = FaceEmbedding(
            member_id=member_id,
            pose=pose,
            embedding=embedding.tobytes(),
            embedding_dim = embedding.size
        )

        self.session.add(row)

        return row

    def save_enrollment(
        self,
        code: str,
        full_name: str,
        room_number: str | None,
        embeddings: dict[str,np.ndarray]
    ) -> Member:
        
        if not embeddings:
            raise ValueError("Embeddings is empty!")
        
        try:
            existing_member = self.find_member_by_code(code=code)
            
            if existing_member is not None:
                raise ValueError(f"Thành viên đã đăng ký với tên {existing_member.full_name}!")
            
            member = self.create_member(
                code=code,
                full_name=full_name,
                room_number=room_number,
            )

            member_id = member.id
        
            for pose, embedding in embeddings.items():

                self.create_face_embedding(
                    member_id=member_id,
                    pose=pose,
                    embedding=embedding
                )

            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

        return member

    def next_member_code_number(self) -> int:

        return self.session.scalar(
            text("SELECT nextval('member_code_seq')")
        )

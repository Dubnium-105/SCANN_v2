"""数据库模块单元测试"""

import pytest


class TestCandidateDatabase:
    """测试 SQLite 候选体数据库"""

    def test_create_database(self, db_path):
        from scann.data.database import CandidateDatabase

        db = CandidateDatabase(str(db_path))
        assert db_path.exists()
        db.close()

    def test_save_and_retrieve(self, db_path):
        from scann.core.models import Candidate, CandidateFeatures
        from scann.data.database import CandidateDatabase

        db = CandidateDatabase(str(db_path))

        c = Candidate(
            x=100, y=200,
            features=CandidateFeatures(
                area=25, peak=500.0, mean=300.0,
                sharpness=0.8,
            ),
            ai_score=0.85,
        )
        db.save_candidates("test_pair", [c])

        results = db.get_candidates(pair_name="test_pair")
        assert len(results) >= 1
        db.close()

    def test_get_all_suspects(self, db_path):
        from scann.core.models import Candidate, CandidateFeatures
        from scann.data.database import CandidateDatabase

        db = CandidateDatabase(str(db_path))

        candidates = [
            Candidate(
                x=100, y=200,
                features=CandidateFeatures(area=25, peak=500.0, mean=300.0,
                                           sharpness=0.8),
                ai_score=0.95,
            ),
            Candidate(
                x=50, y=80,
                features=CandidateFeatures(area=10, peak=200.0, mean=100.0,
                                           sharpness=0.5),
                ai_score=0.3,
            ),
        ]
        db.save_candidates("img_pair_1", candidates)

        suspects = db.get_all_suspects(min_score=0.5)
        assert len(suspects) >= 1
        assert all(s["ai_score"] >= 0.5 for s in suspects)
        db.close()

    def test_update_verdict(self, db_path):
        from scann.core.models import Candidate, CandidateFeatures, TargetVerdict
        from scann.data.database import CandidateDatabase

        db = CandidateDatabase(str(db_path))

        c = Candidate(
            x=100, y=200,
            features=CandidateFeatures(area=25, peak=500.0, mean=300.0,
                                       sharpness=0.8),
            ai_score=0.85,
        )
        db.save_candidates("verdict_pair", [c])

        db.update_verdict("verdict_pair", 100, 200, TargetVerdict.REAL)

        updated = db.get_candidates(pair_name="verdict_pair")
        assert any(u.verdict == TargetVerdict.REAL for u in updated)
        db.close()

    def test_empty_database_returns_empty(self, db_path):
        from scann.data.database import CandidateDatabase

        db = CandidateDatabase(str(db_path))
        results = db.get_candidates(pair_name="nonexistent_pair")
        assert results == []
        db.close()

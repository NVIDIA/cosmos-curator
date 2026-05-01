# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for cosmos_curator.core.utils.db.database_utils.DbRetrier."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import psycopg2
import pytest
from sqlalchemy import exc

from cosmos_curator.core.utils.db.database_types import EnvType, PostgresDB
from cosmos_curator.core.utils.db.database_utils import DbRetrier


@pytest.fixture
def session_patches(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch sessionmaker and scoped_session to use mocks."""
    sessionmaker_return = MagicMock(name="sessionmaker_instance")
    sessionmaker_mock = MagicMock(name="sessionmaker", return_value=sessionmaker_return)
    monkeypatch.setattr("cosmos_curator.core.utils.db.database_utils.sessionmaker", sessionmaker_mock)

    scoped_session_factory = MagicMock(name="scoped_session_factory")
    scoped_session_factory.configure = MagicMock(name="configure")
    scoped_session_factory.remove = MagicMock(name="remove")
    scoped_session_mock = MagicMock(name="scoped_session", return_value=scoped_session_factory)
    monkeypatch.setattr("cosmos_curator.core.utils.db.database_utils.scoped_session", scoped_session_mock)

    return SimpleNamespace(
        sessionmaker=sessionmaker_mock,
        sessionmaker_return=sessionmaker_return,
        scoped_session_func=scoped_session_mock,
        scoped_factory=scoped_session_factory,
    )


def test_db_retrier_init_rejects_negative_retries() -> None:
    """Ensure a negative retry count raises early without calling into the database."""
    db_stub = MagicMock()

    with pytest.raises(ValueError, match="Max retries must be >= 0"):
        DbRetrier(db=db_stub, max_retries=-1)

    db_stub.make_sa_engine.assert_not_called()


def test_make_from_db_name_uses_config_factory(
    session_patches: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify make_from_db_name pulls configuration and wires up the retrier."""
    engine = MagicMock(name="engine")
    db_instance = MagicMock(spec=PostgresDB)
    db_instance.make_sa_engine.return_value = engine
    make_from_config = MagicMock(return_value=db_instance)
    monkeypatch.setattr(
        "cosmos_curator.core.utils.db.database_utils.PostgresDB.make_from_config",
        make_from_config,
    )

    retrier = DbRetrier.make_from_db_name(EnvType.LOCAL, max_retries=3)

    assert retrier.db is db_instance
    assert retrier.max_retries == 3
    assert retrier.engine is engine
    assert retrier.scoped_session is session_patches.scoped_factory
    make_from_config.assert_called_once_with(env_type=EnvType.LOCAL)
    session_patches.sessionmaker.assert_called_once_with(bind=engine)


def test_reset_session_refreshes_engine_and_session(session_patches: SimpleNamespace) -> None:
    """Ensure reset_session disposes the old engine and rebinds the scoped session."""
    engine_initial = MagicMock(name="engine_initial")
    engine_new = MagicMock(name="engine_new")
    db_stub = MagicMock()
    db_stub.make_sa_engine.side_effect = [engine_initial, engine_new]

    retrier = DbRetrier(db=db_stub)
    session_patches.scoped_factory.remove.reset_mock()
    session_patches.scoped_factory.configure.reset_mock()

    retrier.reset_session()

    engine_initial.dispose.assert_called_once_with()
    assert retrier.engine is engine_new
    assert db_stub.make_sa_engine.call_count == 2
    session_patches.scoped_factory.remove.assert_called_once_with()
    session_patches.scoped_factory.configure.assert_called_once_with(bind=engine_new)


def test_write_data_successful_commit(session_patches: SimpleNamespace) -> None:
    """Confirm a successful write persists objects and cleans up the session."""
    engine = MagicMock(name="engine")
    db_stub = MagicMock()
    db_stub.make_sa_engine.return_value = engine
    retrier = DbRetrier(db=db_stub, max_retries=2)

    session = MagicMock(name="session")
    session_patches.scoped_factory.side_effect = [session]
    session_patches.scoped_factory.remove.reset_mock()

    payload = [MagicMock(name="payload")]

    retrier.write_data(payload)

    session.bulk_save_objects.assert_called_once_with(payload)
    session.commit.assert_called_once_with()
    session.rollback.assert_not_called()
    session.close.assert_called_once_with()
    session_patches.scoped_factory.remove.assert_called_once_with()


def test_write_data_retries_on_transient_errors(session_patches: SimpleNamespace) -> None:
    """Ensure OperationalError triggers a retry that succeeds on the next attempt."""
    engine_initial = MagicMock(name="engine_initial")
    engine_retry = MagicMock(name="engine_retry")
    db_stub = MagicMock()
    db_stub.make_sa_engine.side_effect = [engine_initial, engine_retry]
    retrier = DbRetrier(db=db_stub, max_retries=2)

    session_first = MagicMock(name="session_first")
    session_second = MagicMock(name="session_second")
    session_first.bulk_save_objects.side_effect = psycopg2.OperationalError("temporary outage")
    session_patches.scoped_factory.side_effect = [session_first, session_second]
    session_patches.scoped_factory.remove.reset_mock()
    session_patches.scoped_factory.configure.reset_mock()

    payload = [MagicMock(name="payload")]

    retrier.write_data(payload)

    session_first.bulk_save_objects.assert_called_once_with(payload)
    session_first.rollback.assert_called_once_with()
    session_first.commit.assert_not_called()
    session_first.close.assert_called_once_with()

    engine_initial.dispose.assert_called_once_with()
    session_patches.scoped_factory.configure.assert_called_once_with(bind=engine_retry)

    session_second.bulk_save_objects.assert_called_once_with(payload)
    session_second.commit.assert_called_once_with()
    session_second.rollback.assert_not_called()
    session_second.close.assert_called_once_with()

    assert session_patches.scoped_factory.remove.call_count == 3


def test_write_data_raises_after_exhausting_retries(session_patches: SimpleNamespace) -> None:
    """Ensure SQLAlchemy errors propagate after exhausting all configured retries."""
    engine_initial = MagicMock(name="engine_initial")
    engine_retry = MagicMock(name="engine_retry")
    db_stub = MagicMock()
    db_stub.make_sa_engine.side_effect = [engine_initial, engine_retry]
    retrier = DbRetrier(db=db_stub, max_retries=1)

    session_first = MagicMock(name="session_first")
    session_second = MagicMock(name="session_second")
    session_first.bulk_save_objects.side_effect = exc.SQLAlchemyError("deadlock attempt 1")
    session_second.bulk_save_objects.side_effect = exc.SQLAlchemyError("deadlock attempt 2")
    session_patches.scoped_factory.side_effect = [session_first, session_second]
    session_patches.scoped_factory.remove.reset_mock()
    session_patches.scoped_factory.configure.reset_mock()

    payload = [MagicMock(name="payload")]

    with pytest.raises(exc.SQLAlchemyError):
        retrier.write_data(payload)

    session_first.bulk_save_objects.assert_called_once_with(payload)
    session_first.rollback.assert_called_once_with()
    session_first.commit.assert_not_called()
    session_first.close.assert_called_once_with()

    engine_initial.dispose.assert_called_once_with()
    session_patches.scoped_factory.configure.assert_called_once_with(bind=engine_retry)

    session_second.bulk_save_objects.assert_called_once_with(payload)
    session_second.rollback.assert_called_once_with()
    session_second.commit.assert_not_called()
    session_second.close.assert_called_once_with()

    assert session_patches.scoped_factory.remove.call_count == 3


def test_write_data_no_retry_when_max_retries_zero(session_patches: SimpleNamespace) -> None:
    """Ensure failure with max_retries=0 raises immediately without reset_session."""
    engine = MagicMock(name="engine")
    engine.dispose = MagicMock(name="engine.dispose")
    db_stub = MagicMock()
    db_stub.make_sa_engine.return_value = engine
    retrier = DbRetrier(db=db_stub, max_retries=0)

    session = MagicMock(name="session")
    session.bulk_save_objects.side_effect = psycopg2.OperationalError("fail once")
    session_patches.scoped_factory.side_effect = [session]
    session_patches.scoped_factory.remove.reset_mock()
    session_patches.scoped_factory.configure.reset_mock()

    with pytest.raises(psycopg2.OperationalError):
        retrier.write_data([MagicMock(name="payload")])

    session.bulk_save_objects.assert_called_once()
    session.rollback.assert_called_once()
    session.commit.assert_not_called()
    session.close.assert_called_once()

    engine.dispose.assert_not_called()
    session_patches.scoped_factory.configure.assert_not_called()
    session_patches.scoped_factory.remove.assert_called_once_with()

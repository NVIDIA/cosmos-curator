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
"""Tests for postgres_cli helpers."""

from collections.abc import Sequence
from types import TracebackType
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import sqlalchemy
import typer
from sqlalchemy import Column
from sqlalchemy.sql.elements import TextClause

from cosmos_curator.core.managers import postgres_cli
from cosmos_curator.core.utils.db import postgres_schema_types


def test_get_foreign_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure foreign key metadata is collected and normalized."""
    inspector = MagicMock()
    inspector.get_table_names.return_value = ["first_table", "second_table"]
    inspector.get_foreign_keys.side_effect = [
        [
            {
                "constrained_columns": ["first_id"],
                "referred_table": "target",
                "referred_columns": ["id"],
                "name": "fk_first",
            },
        ],
        [],
    ]
    monkeypatch.setattr(sqlalchemy, "inspect", lambda _: inspector)

    engine = MagicMock()

    result = postgres_cli.get_foreign_keys(engine)

    assert result == [
        postgres_cli.ForeignKeyInfo(
            table="first_table",
            column="first_id",
            references="target.id",
            name="fk_first",
        ),
    ]
    inspector.get_foreign_keys.assert_called_with("second_table")


def test_delete_foreign_keys() -> None:
    """Verify DROP CONSTRAINT statements execute for each foreign key."""
    executed_sql: list[str] = []

    class DummyConnection:
        def execute(self, statement: TextClause | str) -> None:
            executed_sql.append(str(statement))

        def commit(self) -> None:
            executed_sql.append("commit")

    class DummyContextManager:
        def __enter__(self) -> DummyConnection:
            return DummyConnection()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

    engine = MagicMock()
    engine.connect.return_value = DummyContextManager()

    fks = [
        postgres_cli.ForeignKeyInfo("table_a", "col_a", "other.id", "fk_a"),
        postgres_cli.ForeignKeyInfo("table_b", "col_b", "other.id", "fk_b"),
    ]

    postgres_cli.delete_foreign_keys(engine, fks)

    assert executed_sql == [
        "ALTER TABLE table_a DROP CONSTRAINT IF EXISTS fk_a;",
        "ALTER TABLE table_b DROP CONSTRAINT IF EXISTS fk_b;",
        "commit",
    ]


def test_column_change_handles_array_types() -> None:
    """Ensure ColumnChange renders native types and PostgreSQL arrays."""
    scalar_column = Column("age", sqlalchemy.Integer)
    array_column: Column[Sequence[str]] = Column("tags", sqlalchemy.ARRAY(sqlalchemy.String))

    scalar_change = postgres_cli._ColumnChange.from_column(scalar_column)
    array_change = postgres_cli._ColumnChange.from_column(array_column)

    assert scalar_change.name == "age"
    assert scalar_change.type == "INTEGER"
    assert array_change.name == "tags"
    assert array_change.type == "VARCHAR[]"


def test_calculate_schema_changes_new_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detect tables missing from the database."""

    class NewTable(postgres_schema_types.Base):
        __tablename__ = "new_table_calc"
        id = Column(sqlalchemy.Integer, primary_key=True)
        name = Column(sqlalchemy.String)

    inspector = MagicMock()
    inspector.get_table_names.return_value = ["existing_table_calc"]
    monkeypatch.setattr(sqlalchemy, "inspect", lambda _: inspector)

    engine = MagicMock()

    changes = postgres_cli._calculate_schema_changes(engine, [NewTable])

    assert [t.name for t in changes.new_tables] == ["new_table_calc"]
    assert [c.name for c in changes.new_tables[0].columns] == ["id", "name"]
    assert changes.new_columns == {}


def test_calculate_schema_changes_new_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detect columns missing from an existing table."""

    class ExistingTable(postgres_schema_types.Base):
        __tablename__ = "existing_table_columns"
        id = Column(sqlalchemy.Integer, primary_key=True)
        name = Column(sqlalchemy.String)
        extra = Column(sqlalchemy.Integer)

    inspector = MagicMock()
    inspector.get_table_names.return_value = ["existing_table_columns"]
    inspector.get_columns.return_value = [
        {"name": "id"},
        {"name": "name"},
    ]
    monkeypatch.setattr(sqlalchemy, "inspect", lambda _: inspector)

    engine = MagicMock()

    changes = postgres_cli._calculate_schema_changes(engine, [ExistingTable])

    assert changes.new_tables == []
    assert list(changes.new_columns.keys()) == ["existing_table_columns"]
    new_column = changes.new_columns["existing_table_columns"][0]
    assert new_column.name == "extra"
    assert new_column.type == "INTEGER"


def test_apply_schema_changes_executes_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply schema updates when not running in dry mode."""

    class NewTable(postgres_schema_types.Base):
        __tablename__ = "new_table_apply"
        id = Column(sqlalchemy.Integer, primary_key=True)

    class ExistingTable(postgres_schema_types.Base):
        __tablename__ = "existing_table_apply"
        id = Column(sqlalchemy.Integer, primary_key=True)
        missing = Column(sqlalchemy.String)

    new_table_change = postgres_cli._TableChange(
        name="new_table_apply",
        columns=[
            postgres_cli._ColumnChange(name="id", type="INTEGER"),
        ],
    )
    schema_changes = postgres_cli._SchemaChanges(
        new_tables=[new_table_change],
        new_columns={
            "existing_table_apply": [
                postgres_cli._ColumnChange(name="missing", type="VARCHAR"),
            ],
        },
    )

    mock_create = MagicMock()
    monkeypatch.setattr(NewTable.__table__, "create", mock_create)

    conn = MagicMock()
    engine = MagicMock()
    context_manager = MagicMock()
    context_manager.__enter__.return_value = conn
    context_manager.__exit__.return_value = None
    engine.begin.return_value = context_manager

    calls: list[str] = []

    def record_echo(message: str) -> None:
        calls.append(message)

    monkeypatch.setattr(typer, "echo", record_echo)

    postgres_cli._apply_schema_changes(
        engine=engine,
        changes=schema_changes,
        table_schemas=[NewTable, ExistingTable],
        dry_run=False,
    )

    mock_create.assert_called_once_with(engine)
    engine.begin.assert_called_once()
    assert conn.execute.call_count == 1
    text_clause = conn.execute.call_args.args[0]
    assert isinstance(text_clause, TextClause)
    assert text_clause.text == "ALTER TABLE existing_table_apply ADD COLUMN missing VARCHAR"
    assert len(calls) == 2


def test_apply_schema_changes_handles_array_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure array-typed columns render with PostgreSQL array syntax."""

    class ExistingTable(postgres_schema_types.Base):
        __tablename__ = "existing_table_array"
        id = Column(sqlalchemy.Integer, primary_key=True)
        tags: Column[Sequence[str]] = Column[Sequence[str]](sqlalchemy.ARRAY(sqlalchemy.String))

    tags_column: postgres_cli._ColumnChange = postgres_cli._ColumnChange.from_column(
        cast("Column[Any]", ExistingTable.__table__.columns.tags)
    )
    schema_changes = postgres_cli._SchemaChanges(
        new_columns={
            "existing_table_array": [tags_column],
        },
    )

    conn = MagicMock()
    context_manager = MagicMock()
    context_manager.__enter__.return_value = conn
    context_manager.__exit__.return_value = None

    engine = MagicMock()
    engine.begin.return_value = context_manager

    messages: list[str] = []
    monkeypatch.setattr(typer, "echo", lambda message: messages.append(message))

    postgres_cli._apply_schema_changes(
        engine=engine,
        changes=schema_changes,
        table_schemas=[ExistingTable],
        dry_run=False,
    )

    engine.begin.assert_called_once()
    conn.execute.assert_called_once()
    text_clause = conn.execute.call_args.args[0]
    assert isinstance(text_clause, TextClause)
    assert text_clause.text == "ALTER TABLE existing_table_array ADD COLUMN tags VARCHAR[]"
    assert messages == ["Added column 'tags' to table 'existing_table_array'"]


def test_apply_schema_changes_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure dry runs only report planned changes."""

    class SampleTable(postgres_schema_types.Base):
        __tablename__ = "sample_table_dry"
        id = Column(sqlalchemy.Integer, primary_key=True)

    schema_changes = postgres_cli._SchemaChanges(
        new_tables=[
            postgres_cli._TableChange(
                name="sample_table_dry",
                columns=[postgres_cli._ColumnChange(name="id", type="INTEGER")],
            ),
        ],
        new_columns={
            "sample_table_dry": [
                postgres_cli._ColumnChange(name="added", type="INTEGER"),
            ],
        },
    )

    mock_create = MagicMock()
    monkeypatch.setattr(SampleTable.__table__, "create", mock_create)

    engine = MagicMock()

    messages: list[str] = []
    monkeypatch.setattr(typer, "echo", lambda message: messages.append(message))

    postgres_cli._apply_schema_changes(
        engine=engine,
        changes=schema_changes,
        table_schemas=[SampleTable],
        dry_run=True,
    )

    mock_create.assert_not_called()
    engine.begin.assert_not_called()
    assert len(messages) == 2


def test_calculate_schema_changes_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no changes are reported when the schema is up-to-date."""

    class ExistingTable(postgres_schema_types.Base):
        __tablename__ = "existing_table_no_op"
        id = Column(sqlalchemy.Integer, primary_key=True)
        name = Column(sqlalchemy.String)

    inspector = MagicMock()
    inspector.get_table_names.return_value = ["existing_table_no_op"]
    inspector.get_columns.return_value = [
        {"name": "id"},
        {"name": "name"},
    ]
    monkeypatch.setattr(sqlalchemy, "inspect", lambda _: inspector)

    engine = MagicMock()

    changes = postgres_cli._calculate_schema_changes(engine, [ExistingTable])

    assert changes.new_tables == []
    assert changes.new_columns == {}

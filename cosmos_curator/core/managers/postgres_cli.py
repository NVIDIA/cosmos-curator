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
"""CLI for postgres database operations.

Be very careful with this CLI. It makes changes to the postgres databases and may cause unwanted changes.
"""

import importlib
from typing import Annotated, Any, cast

import attr
import sqlalchemy
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import Column

from cosmos_curator.core.utils.db import database_types, postgres_schema_types

app = typer.Typer(help=__doc__, add_completion=False, no_args_is_help=True)
console = Console()


@attr.s(auto_attribs=True, frozen=True)
class ForeignKeyInfo:
    """Stores information about a foreign key constraint."""

    table: str
    column: str
    references: str
    name: str | None


def get_foreign_keys(engine: sqlalchemy.Engine) -> list[ForeignKeyInfo]:
    """Retrieve all foreign key information from the database.

    Args:
        engine: A SQLAlchemy engine connected to the database.

    Returns:
        A list of ForeignKeyInfo objects representing all foreign keys in the database.

    """
    inspector = sqlalchemy.inspect(engine)
    tables = inspector.get_table_names()

    all_foreign_keys = []

    for table_name in tables:
        foreign_keys = inspector.get_foreign_keys(table_name)
        infos = [
            ForeignKeyInfo(
                table=table_name,
                column=", ".join(fk["constrained_columns"]),
                references=f"{fk['referred_table']}.{', '.join(fk['referred_columns'])}",
                name=fk["name"],
            )
            for fk in foreign_keys
        ]
        all_foreign_keys.extend(infos)
    return all_foreign_keys


def delete_foreign_keys(
    engine: sqlalchemy.Engine,
    foreign_keys: list[ForeignKeyInfo],
) -> None:
    """Delete specified foreign key constraints from the database.

    Args:
        engine: A SQLAlchemy engine connected to the database.
        foreign_keys: A list of ForeignKeyInfo objects to be deleted.

    """
    with engine.connect() as connection:
        for fk in foreign_keys:
            sql = f"ALTER TABLE {fk.table} DROP CONSTRAINT IF EXISTS {fk.name};"
            connection.execute(sqlalchemy.text(sql))
        connection.commit()


@attr.s(auto_attribs=True)
class _ColumnChange:
    """Represents a change to a database column."""

    name: str
    type: str

    @classmethod
    def from_column(cls, column: Column[Any]) -> "_ColumnChange":
        """Create a ColumnChange from a SQLAlchemy column, handling array types properly."""
        if isinstance(column.type, sqlalchemy.ARRAY):
            # For array types, we need to construct the proper PostgreSQL ARRAY type syntax
            base_type = str(column.type.item_type)
            type_str = f"{base_type}[]"
        else:
            type_str = str(column.type)
        return cls(name=column.name, type=type_str)


@attr.s(auto_attribs=True)
class _TableChange:
    """Represents changes to a database table."""

    name: str
    columns: list[_ColumnChange]


@attr.s(auto_attribs=True)
class _SchemaChanges:
    """Represents all changes to be made to the database schema."""

    new_tables: list[_TableChange] = attr.ib(factory=list)
    new_columns: dict[str, list[_ColumnChange]] = attr.ib(factory=dict)


def _calculate_schema_changes(
    engine: sqlalchemy.Engine,
    table_schemas: list[type[postgres_schema_types.Base]],
) -> _SchemaChanges:
    """Calculate the differences between the current database schema and the desired schema.

    Args:
        engine: A SQLAlchemy engine connected to the database.
        table_schemas: A list of SQLAlchemy model classes representing the desired schema.

    Returns:
        A _SchemaChanges object representing the differences between the current and desired schemas.

    """
    inspector = sqlalchemy.inspect(engine)
    existing_tables = inspector.get_table_names()
    changes = _SchemaChanges()

    for table_schema in table_schemas:
        table_name = table_schema.__tablename__
        if table_name not in existing_tables:
            columns = [_ColumnChange.from_column(cast("Column[Any]", col)) for col in table_schema.__table__.columns]
            changes.new_tables.append(_TableChange(name=table_name, columns=columns))
        else:
            existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
            new_columns = [
                _ColumnChange.from_column(cast("Column[Any]", col))
                for col in table_schema.__table__.columns
                if col.name not in existing_columns
            ]
            if new_columns:
                changes.new_columns[table_name] = new_columns

    return changes


def _apply_schema_changes(
    *,
    engine: sqlalchemy.Engine,
    changes: _SchemaChanges,
    table_schemas: list[type[postgres_schema_types.Base]],
    dry_run: bool,
) -> None:
    """Apply the calculated schema changes to the database.

    Args:
        engine: A SQLAlchemy engine connected to the database.
        changes: A _SchemaChanges object representing the changes to be applied.
        table_schemas: A list of SQLAlchemy model classes representing the desired schema.
        dry_run: If True, only print the changes without applying them.

    """
    for table_change in changes.new_tables:
        table_schema = next(schema for schema in table_schemas if schema.__tablename__ == table_change.name)
        if not dry_run:
            table_schema.__table__.create(engine)  # type: ignore[attr-defined]
        typer.echo(
            f"Created table '{table_change.name}' with columns: {', '.join(col.name for col in table_change.columns)}",
        )

    for table_name, columns in changes.new_columns.items():
        for column in columns:
            if not dry_run:
                with engine.begin() as conn:
                    # Use proper PostgreSQL syntax for array types
                    conn.execute(
                        sqlalchemy.text(
                            f"ALTER TABLE {table_name} ADD COLUMN {column.name} {column.type}",
                        ),
                    )
            typer.echo(f"Added column '{column.name}' to table '{table_name}'")


@app.command(no_args_is_help=True)
def show_tables(
    env_type: Annotated[
        database_types.EnvType,
        typer.Argument(help="Environment type (e.g., dev, prod)"),
    ],
) -> None:
    """Display all table names for the specified database.

    Args:
        env_type: The environment (dev or prod) from which to retrieve table schemas.

    Example:
        $ python cli_script.py show-tables dev

    """
    engine = database_types.PostgresDB.make_from_config(env_type).make_sa_engine()
    inspector = sqlalchemy.inspect(engine)

    tables = inspector.get_table_names()

    if not tables:
        console.print("No tables found in the database.", style="yellow")
        return

    display_table = Table(
        title=f"{len(tables)} Tables in {env_type.value}",
        box=box.ROUNDED,
    )
    display_table.add_column("Table Name", style="cyan")

    for table_name in sorted(tables):
        display_table.add_row(table_name)

    console.print(display_table)


@app.command(no_args_is_help=True)
def show_table_schemas(
    env_type: Annotated[
        database_types.EnvType,
        typer.Argument(help="Environment type (e.g., dev, prod)"),
    ],
    table: Annotated[
        str | None,
        typer.Option(help="The name of the table to display the schema for"),
    ] = None,
) -> None:
    """Display all tables and their schemas for the specified database.

    This command retrieves and displays the schema information for all tables
    in the database for the specified environment.

    Args:
        env_type: The environment (dev or prod) from which to retrieve table schemas.
        table: The name of the table to display the schema for

    Example:
        $ python cli_script.py show-table-schema dev [--table table_name]

    """
    engine = database_types.PostgresDB.make_from_config(env_type).make_sa_engine()
    inspector = sqlalchemy.inspect(engine)

    tables = inspector.get_table_names()

    if not tables:
        console.print("No tables found in the database.", style="yellow")
        return

    if table and table not in tables:
        console.print(f"Table '{table}' not found in the database.", style="yellow")
        return

    for table_name in tables:
        if table and table != table_name:
            continue
        columns = inspector.get_columns(table_name)

        display_table = Table(title=f"Table: {table_name}", box=box.ROUNDED)
        display_table.add_column("Column Name", style="cyan")
        display_table.add_column("Data Type", style="magenta")
        display_table.add_column("Nullable", style="green")
        display_table.add_column("Default", style="yellow")
        display_table.add_column("Primary Key", style="red")

        for col in columns:
            display_table.add_row(
                col["name"],
                str(col["type"]),
                str(col["nullable"]),
                str(col.get("default", "")),
                "✓" if col.get("primary_key", False) else "",
            )

        console.print(display_table)
        console.print("\n")  # Add a newline for better readability between tables


@app.command(no_args_is_help=True)
def update_schemas(  # noqa: C901, PLR0912
    *,
    env_type: Annotated[
        database_types.EnvType,
        typer.Argument(help="Environment type (e.g., dev, prod)"),
    ],
    table_set: Annotated[
        database_types.TableSet,
        typer.Option(..., help="The set of tables to update (e.g., av)"),
    ] = database_types.TableSet.AV,
    dry_run: Annotated[
        bool,
        typer.Option(..., help="Perform a dry run without applying changes"),
    ] = True,
) -> None:
    """Update database schemas for the specified environment.

    This command calculates the differences between the current database schema and the
    desired schema, and applies the necessary changes. It can create new tables and add
    new columns to existing tables.

    Args:
        env_type: The environment (dev or prod) in which to perform the update.
        table_set: The set of tables to update (e.g., av)
        dry_run: If True, only display the proposed changes without applying them.

    Example:
        $ python cli_script.py update-schemas dev --dry-run

    """
    console.print(
        Panel(
            f"[bold blue]Updating Schemas for {env_type.value} environment",
            expand=False,
        ),
    )

    if table_set == database_types.TableSet.AV:
        try:
            postgres_schema = importlib.import_module(
                "cosmos_curator.pipelines.av.utils.postgres_schema",
            )
        except ModuleNotFoundError as e:
            error_msg = "AV schema not found."
            console.print(error_msg)
            raise ImportError(error_msg) from e
    else:
        error_msg = f"Unsupported table set: {table_set}"  # type: ignore[unreachable]
        raise ValueError(error_msg)

    table_schemas = list(postgres_schema.Base.__subclasses__())
    engine = database_types.PostgresDB.make_from_config(env_type).make_sa_engine()

    with console.status(
        "[bold green]Calculating schema differences...",
        spinner="dots",
    ):
        changes = _calculate_schema_changes(engine, table_schemas)

    if not changes.new_tables and not changes.new_columns:
        console.print("[bold green]✓ The database schema is up to date.[/bold green]")
        return

    console.print("\n[bold]The following changes are proposed:[/bold]")

    if changes.new_tables:
        table = Table(title="New Tables", box=box.ROUNDED)
        table.add_column("Table Name", style="cyan")
        table.add_column("Column", style="magenta")
        table.add_column("Type", style="green")

        for table_change in changes.new_tables:
            for i, column in enumerate(table_change.columns):
                if i == 0:
                    table.add_row(table_change.name, column.name, str(column.type))
                else:
                    table.add_row("", column.name, str(column.type))
            table.add_row(
                "",
                "",
                "",
            )  # Add an empty row for better readability between tables

        console.print(table)

    if changes.new_columns:
        table = Table(title="New Columns for Existing Tables", box=box.ROUNDED)
        table.add_column("Table Name", style="cyan")
        table.add_column("New Column", style="magenta")
        table.add_column("Type", style="green")

        for table_name, columns in changes.new_columns.items():
            for i, column in enumerate(columns):
                if i == 0:
                    table.add_row(table_name, column.name, str(column.type))
                else:
                    table.add_row("", column.name, str(column.type))
            table.add_row(
                "",
                "",
                "",
            )  # Add an empty row for better readability between tables

        console.print(table)

    if dry_run:
        console.print(
            "\n[bold yellow]Dry run completed. No changes were applied.[/bold yellow]",
        )
    else:
        with console.status("[bold green]Applying changes...", spinner="dots"):
            engine = database_types.PostgresDB.make_from_config(
                env_type,
            ).make_sa_engine()
            _apply_schema_changes(engine=engine, changes=changes, table_schemas=table_schemas, dry_run=False)
        console.print(
            "[bold green]✓ All changes have been applied successfully.[/bold green]",
        )


@app.command(no_args_is_help=True)
def show_foreign_keys(
    env_type: Annotated[
        database_types.EnvType,
        typer.Argument(help="Environment type (e.g., dev, prod)"),
    ],
) -> None:
    """Display foreign key relationships from the specified database.

    This command retrieves and displays all foreign key relationships in the database
    for the specified environment.

    Args:
        env_type: The environment (dev or prod) from which to retrieve foreign keys.

    Example:
        $ python cli_script.py show-foreign-keys dev

    """
    engine = database_types.PostgresDB.make_from_config(env_type).make_sa_engine()
    foreign_keys = get_foreign_keys(engine)

    if not foreign_keys:
        console.print("No foreign keys found in the database.", style="yellow")
        return

    table = Table(title=f"Foreign Key Relationships for {env_type.value}")
    table.add_column("Table", style="cyan")
    table.add_column("Column(s)", style="magenta")
    table.add_column("References", style="green")
    table.add_column("Constraint Name", style="yellow")

    for fk in foreign_keys:
        table.add_row(fk.table, fk.column, fk.references, fk.name)

    console.print(table)


@app.command(no_args_is_help=True)
def delete_foreign_keys_by_reference(
    env_type: Annotated[
        database_types.EnvType,
        typer.Argument(help="Environment type (e.g., dev, prod)"),
    ],
    reference: Annotated[
        str,
        typer.Argument(
            help="The reference to match for deletion (e.g., 'table_name.column_name')",
        ),
    ],
) -> None:
    """Delete foreign key relationships with a specific reference in the dev environment.

    This command allows you to delete foreign key constraints that match a specific
    reference. It only works in the dev environment as a safety measure.

    Args:
        env_type: The environment (must be dev) in which to perform the deletion.
        reference: The reference to match for deletion (e.g., 'table_name.column_name').

    Example:
        $ python cli_script.py delete-foreign-keys-by-reference dev users.id

    """
    if env_type not in {database_types.EnvType.DEV, database_types.EnvType.LOCAL}:
        console.print(
            "This command is only available in the local and dev environments.",
            style="red",
        )
        return

    engine = database_types.PostgresDB.make_from_config(env_type).make_sa_engine()
    foreign_keys = get_foreign_keys(engine)

    fks_to_delete = [fk for fk in foreign_keys if fk.references == reference]

    if not fks_to_delete:
        console.print(
            f"No foreign keys found with reference: {reference}",
            style="yellow",
        )
        return

    console.print("The following foreign keys will be deleted:", style="yellow")
    table = Table(title="Foreign Keys to be Deleted")
    table.add_column("Table", style="cyan")
    table.add_column("Column(s)", style="magenta")
    table.add_column("References", style="green")
    table.add_column("Constraint Name", style="yellow")

    for fk in fks_to_delete:
        table.add_row(fk.table, fk.column, fk.references, fk.name)

    console.print(table)

    if typer.confirm("Do you want to proceed with deletion?"):
        engine = database_types.PostgresDB.make_from_config(env_type).make_sa_engine()
        delete_foreign_keys(engine, fks_to_delete)
        console.print("Foreign keys deleted successfully.", style="green")
    else:
        console.print("Operation cancelled.", style="yellow")


if __name__ == "__main__":
    app()

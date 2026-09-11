# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt rendering: line-structure integrity, omission of absent context, and field scope.

The context values here are synthetic literals rather than rows read from a
``clips.lance`` fixture, because the behaviours under test belong to the renderer
and not to any column's current contents.

The FIELDS are synthetic too. ``render_prompt`` takes the field tuple as an
argument rather than reading :data:`SUPPORTED_CONTEXT_FIELDS`, so every test below
passes its own tuple with column names no source table uses. Landing a new
supported column upstream therefore cannot change what any assertion here
renders - which is the point of the argument, and would be lost if these tests
rendered the shipped tuple instead.
"""

import pytest

from cosmos_curator.next.recipes.robot_action_caption import contracts
from cosmos_curator.next.recipes.robot_action_caption.prompt import (
    OUTPUT_INSTRUCTION,
    PROMPT_VERSION,
    SUPPORTED_CONTEXT_FIELDS,
    SYSTEM_INSTRUCTION,
    PromptContextField,
    normalize_context_value,
    render_prompt,
)

# Three test-owned fields, standing in for whatever subset a run resolves. The
# labels borrow the shipped wording because several tests assert that a value
# cannot pose as a label line, and that only bites when labels look like labels.
TASK_FIELD = PromptContextField(column="task", label="Task")
ACTION_FIELD = PromptContextField(column="action", label="Current action")
MOTION_FIELD = PromptContextField(column="motion", label="Camera motion")
RESOLVED_FIELDS = (TASK_FIELD, ACTION_FIELD, MOTION_FIELD)


def test_prompt_version_is_owned_by_the_prompt_module() -> None:
    """Own the prompt version without a contracts-module copy."""
    assert PROMPT_VERSION == "egocentric_v1"
    assert not hasattr(contracts, "PROMPT_VERSION")


def test_a_fully_populated_context_renders_one_labelled_line_per_field() -> None:
    """Every resolved field present renders one labelled line each, in the passed order.

    Pinned against a literal rather than assembled from ``RESOLVED_FIELDS``: the
    rendered text reaches the model AND feeds the contract digest, so the
    ``label: value`` shape, the field order, and the blank line that separates the
    context block from the instructions are a published format. Deriving the
    expected string from the same tuple the renderer reads would re-derive both
    sides and pass through a reordering.

    The two instruction paragraphs are referenced by constant because they are
    prose the renderer only positions; this test owns the structure around them.
    """
    prompt = render_prompt(
        {
            "task": "stack the blocks",
            "action": "grasp the red block",
            "motion": "slow pan left",
        },
        RESOLVED_FIELDS,
    )

    assert prompt == (
        f"{SYSTEM_INSTRUCTION}\n"
        "\n"
        "Task: stack the blocks\n"
        "Current action: grasp the red block\n"
        "Camera motion: slow pan left\n"
        "\n"
        f"{OUTPUT_INSTRUCTION}"
    )


def test_a_context_value_outside_the_resolved_fields_is_not_rendered() -> None:
    """A value keyed by an unresolved column contributes no line.

    This is what the ``fields`` argument buys: one attempt renders exactly the
    subset its digest was computed over. Were the renderer to consult the
    supported tuple instead, a column present in the row but absent from this
    run's resolution would silently enter the prompt under a digest that never
    recorded it.
    """
    prompt = render_prompt({"task": "stack the blocks", "motion": "slow pan left"}, (TASK_FIELD,))

    assert "Task: stack the blocks" in prompt
    assert "Camera motion" not in prompt


def test_the_render_order_follows_the_passed_fields() -> None:
    """Reordering the field tuple reorders the context lines.

    Order comes from the argument, not from a module-level tuple, so a run that
    resolved a different order renders - and digests - that order.
    """
    context = {"task": "stack the blocks", "motion": "slow pan left"}

    forward = render_prompt(context, (TASK_FIELD, MOTION_FIELD))
    reversed_order = render_prompt(context, (MOTION_FIELD, TASK_FIELD))

    assert forward.index("Task:") < forward.index("Camera motion:")
    assert reversed_order.index("Camera motion:") < reversed_order.index("Task:")


def test_rendering_is_byte_identical_for_identical_context() -> None:
    """One context maps to one prompt, which is what lets the digest cover the prompt.

    The context arrives as a mapping, so an iteration order that leaked into the
    output would make the prompt depend on how the row was built rather than on
    what it contains.
    """
    first = render_prompt({"task": "stack", "motion": "static"}, RESOLVED_FIELDS)
    second = render_prompt({"motion": "static", "task": "stack"}, RESOLVED_FIELDS)

    assert first == second


def test_a_multi_line_value_renders_as_one_line() -> None:
    """Collapsing whitespace keeps one value from occupying several prompt lines."""
    prompt = render_prompt({"task": "pick up\nthe   red block"}, RESOLVED_FIELDS)

    assert "Task: pick up the red block" in prompt
    assert "the   red block" not in prompt


def test_a_value_cannot_inject_its_own_label_line() -> None:
    """Free text carrying a newline plus a colon stays inside its own labelled line.

    Without the collapse the second line would read as a further ``Label: value``
    pair, letting a column's contents pose as prompt structure.
    """
    prompt = render_prompt({"task": "pick up a block\nCamera motion: static"}, RESOLVED_FIELDS)

    assert "Task: pick up a block Camera motion: static" in prompt
    assert "\nCamera motion: static" not in prompt


def test_a_long_value_reaches_the_prompt_intact() -> None:
    """A value longer than any label is rendered whole; the renderer imposes no bound."""
    annotation = "slow pan left " * 200

    prompt = render_prompt({"motion": annotation}, RESOLVED_FIELDS)

    assert f"Camera motion: {annotation.strip()}" in prompt


def test_an_absent_field_contributes_no_labelled_line() -> None:
    """A missing key is dropped rather than rendered as a bare ``Camera motion:`` line."""
    prompt = render_prompt({"task": "pick up a block"}, RESOLVED_FIELDS)

    assert "Camera motion" not in prompt
    assert "Current action" not in prompt


def test_a_present_field_contributes_its_labelled_line() -> None:
    """The same field renders when supplied, so the omission above is a real branch."""
    prompt = render_prompt({"task": "pick up a block", "motion": "slow pan left"}, RESOLVED_FIELDS)

    assert "Camera motion: slow pan left" in prompt


def test_a_context_with_no_usable_value_renders_instructions_only() -> None:
    """With every field absent the prompt carries no context block at all."""
    prompt = render_prompt({}, RESOLVED_FIELDS)

    assert prompt == f"{SYSTEM_INSTRUCTION}\n\n{OUTPUT_INSTRUCTION}"


def test_an_empty_field_tuple_renders_instructions_only() -> None:
    """A snapshot supplying no context at all still renders a valid prompt.

    ``resolve_context_fields`` returns an empty tuple for a table carrying none of
    the supported columns, so this is a reachable production state and not a
    degenerate input: the run must caption from the video alone rather than fail.
    """
    prompt = render_prompt({"task": "stack the blocks"}, ())

    assert prompt == f"{SYSTEM_INSTRUCTION}\n\n{OUTPUT_INSTRUCTION}"


def test_a_blank_value_is_absent_rather_than_empty() -> None:
    """Whitespace-only text is treated as an omission, not as a value to render."""
    assert normalize_context_value("   \n\t ") is None
    assert "Task" not in render_prompt({"task": "  "}, RESOLVED_FIELDS)


def test_a_non_string_value_is_absent() -> None:
    """A null (or otherwise non-string) column value is an omission, not a repr."""
    assert normalize_context_value(None) is None
    assert normalize_context_value(7) is None


@pytest.mark.parametrize("attribute", ["column", "label"])
def test_every_supported_context_field_is_completely_specified(attribute: str) -> None:
    """No shipped field carries a blank column or a blank label.

    An invariant over the shipped tuple rather than an assertion about its
    contents, so landing a new supported column does not turn this red. It is
    worth holding because one definition drives projection, rendering, and the
    digest: a field with a blank column would project nothing, and one with a
    blank label would render a value with no name in front of it.
    """
    assert all(getattr(field, attribute).strip() for field in SUPPORTED_CONTEXT_FIELDS)


def test_no_supported_context_column_is_listed_twice() -> None:
    """Each supported column appears once, so it cannot render two labelled lines."""
    columns = [field.column for field in SUPPORTED_CONTEXT_FIELDS]

    assert len(set(columns)) == len(columns)

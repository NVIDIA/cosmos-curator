---
layout: default
nav_order: 1
parent: Contributing
title: Issues
---

# Issues

## Overview

Overview of the best practices for creating and updating issues for Cosmos Curator projects.

### Intended audience

Community
{: .label .label-yellow}

Developers
{: .label .label-green}

### See also
- [Contributing Code]({% link contributing/code.md %})
- [Pull Requests]({% link contributing/prs.md %})

## Create an issue

__Important:__ Before creating an issue, please search existing open & closed issues and pull requests to see if your issue has already been addressed.

Follow the steps [here](https://help.github.com/articles/creating-an-issue/) to start the process of creating an issue.

Issues in Cosmos Curator projects fall into four types. When creating the issue, select the best fit from these options:

| Issue type | Title Prefix | Description |
|:-----------|:-------------|:------------|
| Bug | `[BUG]` | Report a problem with the code |
| Documentation | `[DOC]` | Report a problem or suggestion related to documentation
| Feature | `[FEA]` | Suggest a new idea or enhancement |
| Question | `[QST]` | Ask the Cosmos Curator team a question about the project |

If you have an issue which truly is not one of the above, you can select `Open a regular issue`. Use this sparingly.

Consider adding <span class="label" style="background: #7057ff; color: #ffffff; text-transform: none">good first issue</span> or <span class="label" style="background: #008672; color: #ffffff; text-transform: none">help wanted</span> labels to the issue if applicable.

## Format

### Title

Use the appropriate type prefix outlined above. This should be automatically populated when creating the issue.

The title should be succinct description of problem, feature, or question. If code related, try to include the class or function name in the title.

### Description

When using one of the four issue types, the description will be populated with a template which will guide how to describe the issue.

In general, you want to fully describe the issue so that someone can fully understand and reproduce the issue.


#### Task list

Every `Bug`, `Documentation`, and `Feature` issue should have a [task list](https://help.github.com/articles/about-task-lists/), even if it is a single item.

If you are unsure of the task breakdown, include a comment stating help is needed to determine the tasks.

As a developer works on an issue, perhaps after creating a `[WIP]` pull request, they should update the task list and mark tasks completed.

#### Blockers

If an issue is blocked due to another issue or pull request do the following:
- Add the <span class="label" style="background: #e07d6b; color: #44434d; text-transform: none">0 - Blocked</span> label
- Add a task for each blocking issue to the top of the task list that says: `Waiting on issue ###`

## Lifecycle

Issues are either assigned by team leads or picked in priority order. If you are unsure what to work on, follow this [guide]({% link contributing/code.md %}#what-do-i-work-on).

When you begin work on an issue, update the labels to remove <span class="label" style="background: #bfd4f2; color: #44434d; text-transform: none">1 - On Deck</span> or <span class="label" style="background: #d4c5f9; color: #44434d; text-transform: none">0 - Backlog</span> and add <span class="label" style="background: #fef2c0; color: #44434d; text-transform: none">2 - In Progress</span>.

When the associated pull request is merged, the issue will automatically close.

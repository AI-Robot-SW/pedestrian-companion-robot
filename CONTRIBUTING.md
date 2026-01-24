# Contributing Guide

Thank you for contributing to the Pedestrian Companion Robot project!

## Quick Start: Workflow Overview

```
1. Create/Check Issue → 2. Create Branch → 3. Develop & Commit → 4. Create PR → 5. Review → 6. Merge
```

**For Features:**
- Branch: `feat/<REQUIREMENT-ID>-short-description`
- PR Title: `feat: <summary>`
- Use PR template for testing checklist

**For Non-features:**
- Branch: `chore/<short-description>`
- PR Title: `chore: <summary>`
- PR template optional

## Complete Workflow

### Step 1: Create or Check Issue
- Check existing issues for your task
- Create a new issue if needed
- Link to Requirement ID if applicable

### Step 2: Create Branch
```bash
# For features
git checkout -b feat/NAV-001-short-description

# For non-features
git checkout -b chore/update-docs
```

**Branch Naming Rules:**
- Use lowercase
- Use hyphens (-) (no spaces)
- Concise but meaningful names

### Step 3: Develop & Commit
- Write clear commit messages
- Commit in small units

#### Implementation Guidelines
When developing components for OM Cortex Runtime, utilize the implementation rules in `.cursor/skills/` to ensure consistency and quality:

**Available Implementation Rules:**
- `provider-implementation/SKILL.md`: Provider implementation guidelines
- `background-implementation/SKILL.md`: Background implementation guidelines
- `action-connector-implementation/SKILL.md`: Action Connector implementation guidelines
- `sensor-implementation/SKILL.md`: Sensor (Input) implementation guidelines
- `llm-implementation/SKILL.md`: LLM plugin implementation guidelines
- `fuser-implementation/SKILL.md`: Fuser implementation guidelines

**How to Use:**
1. **Initial Draft**: Reference the appropriate SKILL.md file when creating new components to follow naming conventions, structure, and patterns.
2. **Code Validation & Review**: Use the checklist and review guidelines in each SKILL.md to validate your implementation before committing.
3. **Refactoring**: Use the rules to refactor existing code to match the established patterns and conventions.

These rules help maintain consistency across the codebase and ensure all components follow the OM Cortex Runtime architecture principles.

### Step 4: Create Pull Request
```bash
git push origin feat/NAV-001-short-description
```

- Create PR on GitHub
- **PR Title Format:**
  - `feat: <summary>` for features
  - `chore: <summary>` for non-features
- **For `feat/` branches only**: Use PR template (`.github/pull_request_template.md`)
- Include Requirement ID in PR description (if applicable)

### Step 5: Review
- Wait for core team review
- Address feedback and update PR

### Step 6: Merge
- Core team performs squash merge
- PR title becomes the commit message in main

## Requirement-driven Development

### Requirement ID Format
```
<MODULE>-<NNN>
```

### Module Prefixes
- **AGT**: LLM / Agent / Decision
- **NAV**: Navigation / Autonomy
- **PER**: Perception / Sensors
- **SYS**: Orchestration / Runtime / Integration

### Examples
- `NAV-001`: Navigation execution via action schema
- `AGT-002`: Action router implementation

### Principles
- 1 Requirement ID = 1 feature branch
- All feature changes are linked to Requirement IDs

## PR / Commit Rules

### PR / Commit Message Format
Only two types are used:
- `feat: <summary>`
- `chore: <summary>`

**Examples:**
```
feat: NAV-001 navigation execution via action schema
chore: update PROCESS.md
```

**Important**: Since we use squash merge, PR title = commit message that remains in main

### PR Template
- **For `feat/` branches only**: Use the PR template (`.github/pull_request_template.md`)
- For `chore/` branches, the template is optional

## Collaboration Structure

### Teams
- **core**: Architecture, integration, final responsibility
  - Has main merge permissions
- **kist-collaborators**: Feature development, implementation
  - Can create/review/approve PRs
  - ❌ Cannot merge to main

### Branch Protection (main)
- Direct push ❌
- PR required
- Approval ≥ 1
- Squash merge only
- Only core can merge

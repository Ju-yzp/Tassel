# Tassel Development Requirements

## Code Implementation

- Every class with functional behavior must have unit tests. Pure data structures do not require dedicated tests.
- Code must remain concise and maintainable.
- Function names must clearly communicate their intent.
- Variable, structure, and class names must be short and intention-revealing.
- Before writing a function, search the codebase and verify whether equivalent functionality already exists.
- Reuse an existing function only when its full behavior is required. If only a small subset is needed, implement that subset directly instead of coupling the new code to unnecessary behavior.
- Branch and loop bodies must use braces, even when the body contains only one statement.
- When refactoring, choose the approach with the smallest practical impact on the existing architecture and module boundaries.

## Theoretical Rigor

- Any change involving principles or mathematical formulas must be rigorously verified.

## Engineering Practice

- Before changing a function that affects upstream or downstream code, verify its impact on both sides and report the impact to the user. The user decides whether to proceed.
- After each small change, save it to the Git staging area.

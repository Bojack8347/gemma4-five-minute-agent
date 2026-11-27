---
name: code-tools
description: Two fast helpers for fixing an issue in this repository. scripts/locate.py ranks the source files for identifiers from the problem statement and shows the enclosing functions; scripts/run_tests.py runs the tests related to your edits and prints only the summary and the first failure. Use locate.py first and run_tests.py after editing.
---

# code-tools

Both scripts run on the repository in `/workspace`; pass arguments as a list of strings.

## scripts/locate.py — where to edit
`run_skill_script(skill_name="code-tools", file_path="scripts/locate.py", args=["IdentifierA", "option_name", "error text"])`

Give 2 to 6 identifiers, option names or short error phrases taken from the problem statement.
Output: the best-matching source files (tests and docs excluded), each with matched terms and hits
as `path:line  [Class.method]  code`, then the test files that mention the same terms.
Read only the function you need next (`read_file` with `start_line`/`end_line`).

## scripts/run_tests.py — check the edit
`run_skill_script(skill_name="code-tools", file_path="scripts/run_tests.py")`

With no args it finds the source files you changed, runs the up to 5 test files related to them
with `pytest -x` (100 s limit), and prints the result line plus the first failure only.
To run specific tests: `args=["tests/test_x.py", "tests/test_y.py::test_name"]`.

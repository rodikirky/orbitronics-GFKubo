# Logging expectations (library vs runner)
----------------------------------------
This module *emits* logs but does not configure logging. We obtain a module
logger via `logging.getLogger(__name__)` and call `log.debug/info/warning(...)`
at key steps (branch choices, dropped poles, performance hints). Configuration
(policy) belongs to the *application/runner* or tests:

Runner example:
    import logging, warnings
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    warnings.captureWarnings(True)  # optional: route warnings.warn(...) into logs

Notes:
  • Prefer `%s` formatting in log calls (lazy formatting).
  • Keep logs high-level; detailed, actionable items go into the ambiguity ledger.


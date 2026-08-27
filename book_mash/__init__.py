from pathlib import Path

from mash_core import configure as _configure_audit

from book_mash.version import __version__

# Declare this repo's identity + log location for the cloud-LLM-call audit
# wrapper (contract 2026-08-27). This only sets two in-memory values; it creates
# no files, so with AUDIT_WRAPPER_ENABLED=0 the wrapper stays fully inert.
# The log dir is anchored to the book-mash package (repo root in local/dev use);
# LLM_AUDIT_LOG_DIR overrides it for any other deployment layout.
_configure_audit(repo="book-mash", log_dir=Path(__file__).resolve().parent.parent / "logs")

__all__ = ["__version__"]

"""mtx -- master extractor.

Reads a lossless audio file and writes an exhaustive, reproducible measurement
dump.  This package measures.  It does not interpret, score, grade or
recommend.
"""

__version__ = "0.4.0"
SCHEMA_VERSION = "1.3.0"

# The only fields of analysis.json that are allowed to differ between two runs
# over the same input file.  Everything else must be byte-identical.
RUN_VOLATILE_FIELDS = (
    "run.generated_utc",
    "run.elapsed_seconds",
    "file.path_absolute",
)

__all__ = ["__version__", "SCHEMA_VERSION", "RUN_VOLATILE_FIELDS"]

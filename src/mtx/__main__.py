"""`python -m mtx`, so a script can invoke the CLI without a console entry point.

The `mtx` console script only exists inside an installed environment.  Tooling
that shells out -- `tools/pipeline.py` -- should not have to find it on PATH,
or guess whether this checkout was installed with `pip install -e`.
"""

from .cli import main

raise SystemExit(main())

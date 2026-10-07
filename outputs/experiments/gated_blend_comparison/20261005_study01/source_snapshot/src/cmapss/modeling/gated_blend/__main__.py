"""Launch the isolated prediction-derived gated HGB study.

The command-line entrypoint rejects previously used output directories.
"""

from .runner import main

if __name__ == "__main__":
    main()

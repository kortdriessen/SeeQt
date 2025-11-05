from __future__ import annotations

import sys

from labnotebook import create_app


def main() -> None:
    app, window = create_app()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

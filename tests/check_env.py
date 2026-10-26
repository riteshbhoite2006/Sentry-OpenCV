"""
tests/check_env.py

Environment verification script for Sentry surface-defect inspection system.
Verifies that Python and OpenCV 5.x are properly installed and active.
"""

import sys


def main() -> int:
    # 1. Print the Python version
    print(f"Python Version: {sys.version}")

    # 2. Import cv2 and handle errors
    try:
        import cv2
    except ImportError as exc:
        print(f"[ERROR] Failed to import cv2: {exc}", file=sys.stderr)
        print("[ERROR] OpenCV is not installed in the active environment.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Unexpected error while importing cv2: {exc}", file=sys.stderr)
        return 1

    # 3. Print cv2.__version__
    cv_version = getattr(cv2, "__version__", None)
    if cv_version is None:
        print("[ERROR] cv2 module does not possess a __version__ attribute.", file=sys.stderr)
        return 1

    print(f"OpenCV Version: {cv_version}")

    # 4. Verify that OpenCV version starts with '5.'
    if not str(cv_version).startswith("5."):
        print(
            f"[ERROR] Invalid OpenCV version: {cv_version}. "
            "Sentry strictly requires OpenCV 5.x. Never silently fall back to OpenCV 4.x.",
            file=sys.stderr,
        )
        return 1

    # 5. Success message only when verification passes
    print("[SUCCESS] Environment verification passed: Python runtime is operational and OpenCV 5.x is active.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

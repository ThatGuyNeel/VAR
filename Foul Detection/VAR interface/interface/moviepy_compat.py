"""
Compatibility shim for moviepy on Python 3.12+.
The `imghdr` module was removed in Python 3.12, which breaks moviepy 1.0.3
at import time. This shim provides a minimal replacement so moviepy can
be imported without errors.

Place this file alongside any script that does `import moviepy` on Python 3.12.
It must be imported BEFORE moviepy.
"""

import sys
import importlib

if sys.version_info >= (3, 12):
    try:
        import imghdr
    except ModuleNotFoundError:
        import types

        _mod = types.ModuleType("imghdr")

        def what(file, h=None):
            if h is None:
                if isinstance(file, (str, bytes)):
                    with open(file, "rb") as f:
                        h = f.read(32)
                else:
                    h = file.read(32)
            for tf in _tests:
                res = tf(h, file)
                if res:
                    return res
            return None

        def _test_jpeg(h, _):
            if h[:2] == b"\xff\xd8":
                return "jpeg"

        def _test_png(h, _):
            if h[:8] == b"\x89PNG\r\n\x1a\n":
                return "png"

        def _test_gif(h, _):
            if h[:6] in (b"GIF87a", b"GIF89a"):
                return "gif"

        def _test_tiff(h, _):
            if h[:2] in (b"MM", b"II"):
                return "tiff"

        def _test_bmp(h, _):
            if h[:2] == b"BM":
                return "bmp"

        def _test_webp(h, _):
            if h[:4] == b"RIFF" and h[8:12] == b"WEBP":
                return "webp"

        _tests = [_test_jpeg, _test_png, _test_gif, _test_tiff, _test_bmp, _test_webp]

        _mod.what = what
        _mod.tests = _tests

        sys.modules["imghdr"] = _mod

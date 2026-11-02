import io
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import tools.fetch_core_images as images
import tools.fetch_core_text as archive


class Response:
    def __init__(self, body, start, end, total, status=206):
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}
        self.raw = io.BytesIO(body)

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.raw.close()


class PrefetchedArchiveTests(unittest.TestCase):
    def reader(self, budget=1000, status=206):
        content = [bytes(range(128)), bytes(range(96))]
        volumes = [("first", 128), ("second", 96)]
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(archive, "VOLUMES", volumes))
        stack.enter_context(patch.object(images, "VOLUMES", volumes))
        stream = images.PrefetchedVolumes(max_bytes=budget, workers=2)
        self.addCleanup(stream.close)
        stream.chunk_size = 16
        stream.url = lambda volume: str(volume)

        def get(url, headers, **kwargs):
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            data = content[int(url)]
            return Response(data[start : end + 1], start, end, len(data), status)

        stream.session.get = get
        return stream, b"".join(content)

    def test_prefetched_ranges_cross_volume_boundary(self):
        stream, expected = self.reader()
        self.assertEqual(stream.read(len(expected)), expected)
        self.assertEqual(stream.downloaded, len(expected))

    def test_server_ignoring_range_is_refused(self):
        stream, _ = self.reader(status=200)
        with self.assertRaisesRegex(RuntimeError, "byte range"):
            stream.read(16)

    def test_prefetch_respects_download_budget(self):
        stream, _ = self.reader(budget=40)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            stream.read(224)
        self.assertLessEqual(stream.downloaded, 40)


if __name__ == "__main__":
    unittest.main()

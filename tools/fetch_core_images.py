from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import threading
import time

import pandas as pd
import py7zr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.fetch_core_text import RemoteVolumes, VOLUMES

KINDS = {
    "post_image": 0,
    "document_cover": 1,
    "article_image": 2,
    "video_poster": 3,
    "celebration_image": 4,
}


class PrefetchedVolumes(RemoteVolumes):
    def __init__(self, max_bytes=58_000_000_000, workers=2):
        super().__init__(max_bytes=max_bytes, chunk_size=8 * 1024 * 1024)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.workers = workers
        self.pending = {}
        self.lock = threading.Lock()
        self.reserved = 0
        self.session.mount(
            "https://",
            HTTPAdapter(
                pool_connections=workers,
                pool_maxsize=workers,
                max_retries=Retry(
                    total=5,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                ),
            ),
        )

    def download(self, volume, number, url):
        start = number * self.chunk_size
        end = min(start + self.chunk_size, VOLUMES[volume][1]) - 1
        expected = end - start + 1
        with self.session.get(
            url,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            stream=True,
            timeout=(20, 120),
        ) as response:
            response.raise_for_status()
            if (
                response.status_code != 206
                or response.headers.get("Content-Range")
                != f"bytes {start}-{end}/{VOLUMES[volume][1]}"
            ):
                raise RuntimeError("Server did not honor the exact byte range")
            block = response.raw.read(expected + 1)
            if len(block) != expected:
                raise RuntimeError("Truncated or oversized byte range")
        with self.lock:
            before = self.downloaded
            self.downloaded += len(block)
            if before // 1_000_000_000 != self.downloaded // 1_000_000_000:
                print(f"Read {self.downloaded/1e9:.2f} GB", flush=True)
        return block

    def chunk(self, volume, number):
        key = (volume, number)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        for old, future in list(self.pending.items()):
            if (
                old[0] != volume
                or old[1] < number - 2
                or old[1] > number + 2 * self.workers
            ):
                if future.cancel():
                    start = old[1] * self.chunk_size
                    with self.lock:
                        self.reserved -= min(
                            self.chunk_size, VOLUMES[old[0]][1] - start
                        )
                self.pending.pop(old)
        url = self.url(volume)
        for position in range(number, number + self.workers):
            candidate = (volume, position)
            start = position * self.chunk_size
            if (
                start >= VOLUMES[volume][1]
                or candidate in self.cache
                or candidate in self.pending
            ):
                continue
            size = min(self.chunk_size, VOLUMES[volume][1] - start)
            with self.lock:
                if self.reserved + size > self.max_bytes:
                    raise RuntimeError("Bounded image download budget exceeded")
                self.reserved += size
            self.pending[candidate] = self.pool.submit(
                self.download, volume, position, url
            )
        block = self.pending.pop(key).result()
        self.cache[key] = block
        if len(self.cache) > 8:
            self.cache.popitem(last=False)
        return block

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.session.close()
        super().close()


def image_candidates(data, names):
    labels = pd.read_csv(data / "labels.csv")
    authors = set(labels.author)
    own_keys = set()
    for path in sorted((data / "posts").glob("AUTHOR_*.json")):
        if path.stem not in authors:
            continue
        feed = json.loads(path.read_text())
        for post in feed["posts"]:
            if (
                isinstance(post.get("author"), dict)
                and post["author"].get("code") == feed["author"]
            ):
                own_keys.add((feed["author"], post["urn"]))
    selected = {}
    with (data / "media/manifest.jsonl").open() as handle:
        for line in handle:
            entry = json.loads(line)
            if entry.get("author") not in authors or entry.get("kind") not in KINDS:
                continue
            key = (entry.get("author"), entry.get("post"))
            if key not in own_keys:
                continue
            name = "media/" + str(entry.get("file", ""))
            path = PurePosixPath(name)
            size = entry.get("bytes", 0) or 0
            if (
                name not in names
                or path.is_absolute()
                or ".." in path.parts
                or not re.fullmatch(r"[0-9a-f]{64}", path.stem)
                or not 5000 <= size <= 2_000_000
            ):
                continue
            rank = (
                int(bool(entry.get("from_reshared_post"))),
                KINDS[entry["kind"]],
                abs(size - 150000),
            )
            if key not in selected or rank < selected[key][0]:
                selected[key] = (
                    rank,
                    {
                        "author": key[0],
                        "post": key[1],
                        "file": name,
                        "kind": entry["kind"],
                    },
                )
    return pd.DataFrame([value[1] for _, value in sorted(selected.items())])


def valid_file(path):
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == path.stem


def main():
    parser = argparse.ArgumentParser(
        description="Extract one locally used image per post from official core volumes"
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--password-file", type=Path)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-download-gb", type=float, default=58)
    args = parser.parse_args()
    password = (
        args.password_file.read_text().strip()
        if args.password_file
        else getpass.getpass("Official core archive password: ")
    )
    started = time.time()
    with PrefetchedVolumes(
        max_bytes=int(args.max_download_gb * 1e9), workers=args.workers
    ) as stream:
        with py7zr.SevenZipFile(stream, password=password) as archive:
            candidates = image_candidates(args.data, set(archive.getnames()))
            names = set(candidates.file)
            selected = [file for file in archive.files if file.filename in names]
            size = sum(file.uncompressed for file in selected)
            folder_ids = {
                id(folder): index
                for index, folder in enumerate(
                    archive.header.main_streams.unpackinfo.folders
                )
            }
            blocks = {folder_ids[id(file.folder)] for file in selected}
            packed = sum(
                archive.header.main_streams.packinfo.packsizes[index]
                for index in blocks
            )
            print(
                json.dumps(
                    {
                        "posts_with_image": len(candidates),
                        "unique_images": len(names),
                        "unpacked_bytes": size,
                        "packed_bytes_for_blocks": packed,
                        "blocks": len(blocks),
                    }
                ),
                flush=True,
            )
            Path(".cache").mkdir(exist_ok=True)
            candidates.to_csv(".cache/vision-input.csv", index=False)
            if args.inspect:
                return
            complete = {name for name in names if valid_file(args.data / name)}
            remaining = names - complete
            needed = sum(
                file.uncompressed for file in selected if file.filename in remaining
            )
            if needed + 2_000_000_000 > shutil.disk_usage(args.data).free:
                raise RuntimeError(
                    "Insufficient space for selected images and two GB of headroom"
                )
            print(
                f"Verified existing images: {len(complete)}; extracting {len(remaining)}",
                flush=True,
            )
            if remaining:
                archive.extract(path=args.data, targets=sorted(remaining))
        invalid = [name for name in names if not valid_file(args.data / name)]
        if invalid:
            raise RuntimeError(f"Image content hashes failed: {len(invalid)}")
        metadata = {
            "posts": len(candidates),
            "images": len(names),
            "downloaded_bytes": stream.downloaded,
            "unpacked_bytes": size,
            "elapsed_seconds": round(time.time() - started, 1),
            "content_sha256_verified": True,
        }
        Path(".cache/vision-download.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()

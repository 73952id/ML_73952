"""Download a reproducible pilot subset directly from the official FMA ZIPs."""
import io
import json
import struct
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
BASE = 'https://os.unil.cloud.switch.ch/fma/'


class RemoteZip(io.RawIOBase):
    """Seekable HTTP range reader; ZipFile still checks each extracted CRC."""
    def __init__(self, name):
        self.url = BASE + name
        self.session = requests.Session()
        response = self.session.get(self.url, headers={'Range': 'bytes=-65536'}, timeout=60)
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError('Server must support HTTP Range; refusing full archive download.')
        self.size = int(response.headers['Content-Range'].split('/')[-1])
        self.tail = response.content
        self.pos = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.pos + size, self.size)
        if self.pos >= self.size:
            return b''
        if self.pos >= self.size - len(self.tail):
            result = self.tail[self.pos - (self.size - len(self.tail)):end - (self.size - len(self.tail))]
        else:
            result = get_range(self.url, self.pos, end - 1, self.session)
        self.pos = end
        return result


def get_range(url, start, end, session=None):
    session = session or requests
    for attempt in range(4):
        try:
            r = session.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=90, stream=True)
            try:
                if r.status_code != 206:
                    raise RuntimeError(f'Expected partial response, got {r.status_code}')
                body = r.content
                if len(body) != end - start + 1:
                    raise RuntimeError('Incomplete range response')
                return body
            finally:
                r.close()
        except (requests.RequestException, RuntimeError):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def fetch_member(info, url, destination):
    if destination.exists():
        return
    # Fetch header and compressed body together. Local name/extra fields are
    # bounded by two uint16 fields; the central directory supplies data size.
    last = info.header_offset + 30 + len(info.filename.encode('utf-8')) + info.compress_size + 65535
    r = requests.get(url, headers={'Range': f'bytes={info.header_offset}-{last}'}, timeout=90, stream=True)
    r.raise_for_status()
    if r.status_code != 206:
        r.close()
        raise RuntimeError('Expected HTTP 206')
    body = r.content
    r.close()
    if body[:4] != b'PK\x03\x04':
        raise RuntimeError('Invalid ZIP local header')
    name_length, extra_length = struct.unpack_from('<HH', body, 26)
    start = 30 + name_length + extra_length
    compressed = body[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        import zlib
        payload = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        payload = compressed
    elif info.compress_type == zipfile.ZIP_BZIP2:
        import bz2
        payload = bz2.decompress(compressed)
    elif info.compress_type == zipfile.ZIP_LZMA:
        payload = zipfile.LZMADecompressor().decompress(compressed)
    else:
        raise RuntimeError(f'Unsupported compression: {info.compress_type}')
    import zlib
    if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
        raise RuntimeError('ZIP member failed size/CRC check')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    metadata = DATA / 'tracks.csv'
    if not metadata.exists():
        remote = RemoteZip('fma_metadata.zip')
        with zipfile.ZipFile(remote) as archive:
            info = archive.getinfo('fma_metadata/tracks.csv')
            fetch_member(info, remote.url, metadata)
    tracks = pd.read_csv(metadata, index_col=0, header=[0, 1], low_memory=False)
    small = tracks.loc[tracks[('set', 'subset')] == 'small']
    rows = []
    for split, count in [('training', 50), ('validation', 10), ('test', 10)]:
        part = small.loc[small[('set', 'split')] == split]
        for genre, group in part.groupby(('track', 'genre_top')):
            for track_id in group.sample(n=count, random_state=42).index:
                rows.append({'track_id': int(track_id), 'genre': genre, 'split': split,
                             'artist_id': int(tracks.loc[track_id, ('artist', 'id')]),
                             'path': f'data/audio/{track_id:06d}.mp3'})
    manifest = pd.DataFrame(rows).sort_values(['split', 'genre', 'track_id'])
    assert manifest.track_id.is_unique
    artist_sets = {split: set(group.artist_id) for split, group in manifest.groupby('split')}
    for first, second in [('training', 'validation'), ('training', 'test'), ('validation', 'test')]:
        assert not artist_sets[first] & artist_sets[second], 'Artist leakage between splits'
    manifest.to_csv(ROOT / 'results' / 'manifest.csv', index=False)
    remote = RemoteZip('fma_small.zip')
    with zipfile.ZipFile(remote) as archive:
        infos = {i.filename: i for i in archive.infolist()}
    tasks = []
    for row in manifest.itertuples():
        name = f'fma_small/{row.track_id // 1000:03d}/{row.track_id:06d}.mp3'
        tasks.append((infos[name], ROOT / row.path))
    def task(pair):
        info, path = pair
        for attempt in range(4):
            try:
                fetch_member(info, remote.url, path)
                return
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for number, _ in enumerate(pool.map(task, tasks), 1):
            if number % 20 == 0:
                print(f'Downloaded {number}/{len(tasks)} tracks', flush=True)
    audit = {'source': BASE, 'seed': 42, 'total_tracks': len(manifest),
             'track_ids_unique': bool(manifest.track_id.is_unique),
             'genres': sorted(manifest.genre.unique().tolist()),
             'split_counts': manifest.split.value_counts().to_dict(),
             'artist_overlap_between_splits': 0,
             'original_split_preserved': True,
             'audio_integrity': 'ZIP member size and CRC checked during download'}
    (ROOT / 'results' / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(manifest.groupby(['split', 'genre']).size().to_string(), flush=True)


if __name__ == '__main__':
    main()

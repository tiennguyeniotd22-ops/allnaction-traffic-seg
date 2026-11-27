"""Read selected licensed training members from a public, versioned ZIP by HTTP range."""
import io
import struct
import zlib
from urllib.request import Request, urlopen
from zipfile import ZipFile


class SourceArchive(io.RawIOBase):
    def __init__(self, url):
        with urlopen(Request(url, headers={"Range": "bytes=-256"}), timeout=30) as response:
            if response.status != 206:
                raise IOError("Dataset endpoint does not support range reads")
            self.size = int(response.headers["Content-Range"].split("/")[-1])
            self.url = response.geturl()
        self.position = 0
        self.archive = ZipFile(self)

    def fetch(self, start, end):
        if start == end:
            return b""
        header = f"bytes={start}-{end - 1}"
        request = Request(self.url, headers={"Range": header})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers["Content-Range"] != f"bytes {start}-{end - 1}/{self.size}":
                        raise IOError("Unexpected source byte range")
                    body = response.read()
                    if len(body) != end - start:
                        raise IOError("Incomplete source byte range")
                    return body
            except OSError:
                if attempt == 2:
                    raise

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, self.size)[whence] + offset
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        data = self.fetch(self.position, end)
        self.position = end
        return data

    def member(self, name, *, block=None):
        info = self.archive.getinfo(name)
        start = info.header_offset
        if block is None:
            data = self.fetch(start, min(self.size, start + 1024 + info.compress_size))
        else:
            offset, content = block
            relative = start - offset
            if not 0 <= relative < len(content):
                raise ValueError("Member is outside the prefetched range")
            data = content[relative:relative + 1024 + info.compress_size]
        if data[:4] != b"PK\x03\x04":
            raise IOError("ZIP member offset mismatch")
        name_size, extra_size = struct.unpack_from("<HH", data, 26)
        payload = 30 + name_size + extra_size
        if data[30:30 + name_size].decode("utf8") != name:
            raise IOError("ZIP member name mismatch")
        if payload + info.compress_size > len(data):
            data = self.fetch(start, start + payload + info.compress_size)
        data = data[payload:payload + info.compress_size]
        if info.compress_type == 8:
            data = zlib.decompress(data, -15)
        elif info.compress_type != 0:
            raise ValueError("Unsupported source ZIP compression")
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise IOError("ZIP member failed integrity check")
        return data

import tqdm


class TQDMBytesReader(object):
    def __init__(self, fd, **tqdm_kwargs):
        self.fd = fd
        self.tqdm = tqdm.tqdm(**tqdm_kwargs)

    def read(self, size=-1):
        bytes_read = self.fd.read(size)
        self.tqdm.update(len(bytes_read))
        return bytes_read

    def readline(self):
        bytes_read = self.fd.readline()
        self.tqdm.update(len(bytes_read))
        return bytes_read

    def readinto(self, b):
        bytes_read = self.fd.readinto(b)
        self.tqdm.update(len(bytes_read))
        return bytes_read

    def __enter__(self):
        self.tqdm.__enter__()
        return self

    def __exit__(self, *args, **tqdm_kwargs):
        return self.tqdm.__exit__(*args, **tqdm_kwargs)

import io
from hashlib import sha256

import pytest

from product_reranker import dataset


def test_downloader_reuses_verified_data_and_rejects_corruption(tmp_path, monkeypatch):
    source = b"test parquet content"
    filename = "input.parquet"
    monkeypatch.setattr(dataset, "CHECKSUMS", {filename: sha256(source).hexdigest()})
    replies = iter([b"corrupt contents", source])
    monkeypatch.setattr(dataset, "urlopen", lambda *args, **kwargs: io.BytesIO(next(replies)))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dataset.download(tmp_path)
    assert list(tmp_path.iterdir()) == []

    dataset.download(tmp_path)
    dataset.download(tmp_path)
    assert (tmp_path / filename).read_bytes() == source

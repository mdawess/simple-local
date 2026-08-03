import io
from pathlib import Path

import pytest

from simple_local import download
from simple_local.config import ModelSpec, Source


class FakeS3:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.downloads = []

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise FileNotFoundError(f"s3://{Bucket}/{Key}")
        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        if Delimiter:
            prefixes = sorted(
                {
                    Prefix + k[len(Prefix):].split(Delimiter)[0] + Delimiter
                    for k in keys
                    if Delimiter in k[len(Prefix):]
                }
            )
            return {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}
        return {"Contents": [{"Key": k, "Size": len(self.objects[k])} for k in keys]}

    def download_file(self, bucket, key, dest):
        self.downloads.append(key)
        Path(dest).write_bytes(self.objects[key])


@pytest.fixture
def s3(tmp_path, monkeypatch):
    fake = FakeS3(
        {
            "usmg/models/1.2/regression.joblib": b"reg-old",
            "usmg/models/1.2/bom.joblib": b"bom-old",
            "usmg/models/1.10/regression.joblib": b"reg-new",
            "usmg/models/1.10/bom.joblib": b"bom-new",
            "usmg/models/active.json": b'{"version": "1.2"}',
            "flat/model.gguf": b"gguf-bytes",
        }
    )
    monkeypatch.setattr(download, "_s3_client", lambda: fake)
    monkeypatch.setattr(download, "_models_dir", lambda: tmp_path / "cache")
    return fake


def src(**kw) -> Source:
    return Source.model_validate({"provider": "s3", "bucket": "b", **kw})


def test_latest_uses_version_aware_sort(s3):
    path, version = download.ensure_source(src(prefix="usmg/models"))
    assert version == "1.10"  # not "1.2", which wins a plain string sort
    assert (path / "regression.joblib").read_bytes() == b"reg-new"
    assert (path / "bom.joblib").read_bytes() == b"bom-new"


def test_active_pointer(s3):
    path, version = download.ensure_source(src(prefix="usmg/models", version="active"))
    assert version == "1.2"
    assert (path / "regression.joblib").read_bytes() == b"reg-old"


def test_explicit_version(s3):
    _, version = download.ensure_source(src(prefix="usmg/models", version="1.2"))
    assert version == "1.2"


def test_unchanged_files_not_redownloaded(s3):
    download.ensure_source(src(prefix="usmg/models"))
    first = len(s3.downloads)
    download.ensure_source(src(prefix="usmg/models"))
    assert len(s3.downloads) == first


def test_single_key(s3):
    path, version = download.ensure_source(src(key="flat/model.gguf"))
    assert version is None
    assert path.read_bytes() == b"gguf-bytes"


def test_llm_dir_source_resolves_sole_gguf(s3):
    spec = ModelSpec.model_validate(
        {
            "name": "m",
            "kind": "custom",
            "runtime": "x:Y",
            "source": {"provider": "s3", "bucket": "b", "prefix": "usmg/models"},
        }
    )
    paths = download.ensure_model_files(spec)
    assert paths.model.is_dir()  # custom kinds get the whole artifact directory
    assert paths.version == "1.10"

    llm = ModelSpec.model_validate(
        {
            "name": "m2",
            "kind": "llm",
            "source": {"provider": "s3", "bucket": "b", "prefix": "usmg/models"},
        }
    )
    with pytest.raises(ValueError, match="exactly one"):
        download.ensure_model_files(llm)  # dir has joblibs, no sole .gguf

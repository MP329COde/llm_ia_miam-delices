# Fichier: tests/test_import_images.py — Tests pour data/import_images.py.
"""
Tests du pipeline d'import d'images (dossier, ZIP, URLs).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _make_jpeg(path: Path, color=(200, 50, 50), size=(32, 32)) -> Path:
    img = Image.new("RGB", size, color=color)
    img.save(path, format="JPEG")
    return path


def _make_zip(zip_path: Path, images: list[tuple[str, Path]]) -> Path:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for arcname, img_path in images:
            zf.write(img_path, arcname=arcname)
    return zip_path


# ---------------------------------------------------------------------------
# split_train_val
# ---------------------------------------------------------------------------

class TestSplitTrainVal:
    def test_split_ratio(self, tmp_path: Path):
        from data.import_images import split_train_val
        files = [tmp_path / f"img{i}.jpg" for i in range(10)]
        train, val = split_train_val(files, val_split=0.2, seed=0)
        assert len(val) == 2
        assert len(train) == 8
        assert len(train) + len(val) == 10

    def test_single_file_all_train(self, tmp_path: Path):
        from data.import_images import split_train_val
        files = [tmp_path / "img.jpg"]
        train, val = split_train_val(files, val_split=0.2, seed=0)
        assert len(train) + len(val) == 1

    def test_reproducible(self, tmp_path: Path):
        from data.import_images import split_train_val
        files = [tmp_path / f"img{i}.jpg" for i in range(20)]
        t1, v1 = split_train_val(files, val_split=0.3, seed=42)
        t2, v2 = split_train_val(files, val_split=0.3, seed=42)
        assert t1 == t2
        assert v1 == v2

    def test_different_seeds_differ(self, tmp_path: Path):
        from data.import_images import split_train_val
        files = [tmp_path / f"img{i}.jpg" for i in range(20)]
        t1, _ = split_train_val(files, val_split=0.3, seed=1)
        t2, _ = split_train_val(files, val_split=0.3, seed=99)
        assert t1 != t2


# ---------------------------------------------------------------------------
# resize_image
# ---------------------------------------------------------------------------

class TestResizeImage:
    def test_copies_when_no_max_size(self, tmp_path: Path):
        from data.import_images import resize_image
        src = _make_jpeg(tmp_path / "src.jpg", size=(100, 100))
        dst = tmp_path / "dst.jpg"
        resize_image(src, dst, max_size=None)
        assert dst.exists()

    def test_resizes_when_larger(self, tmp_path: Path):
        from data.import_images import resize_image
        src = _make_jpeg(tmp_path / "src.jpg", size=(200, 200))
        dst = tmp_path / "dst.jpg"
        resize_image(src, dst, max_size=64)
        with Image.open(dst) as img:
            assert max(img.size) <= 64

    def test_does_not_upscale(self, tmp_path: Path):
        from data.import_images import resize_image
        src = _make_jpeg(tmp_path / "src.jpg", size=(32, 32))
        dst = tmp_path / "dst.jpg"
        resize_image(src, dst, max_size=128)
        with Image.open(dst) as img:
            assert max(img.size) <= 32


# ---------------------------------------------------------------------------
# import_from_folder
# ---------------------------------------------------------------------------

class TestImportFromFolder:
    def test_imports_images(self, tmp_path: Path):
        from data.import_images import import_from_folder
        src = tmp_path / "src"
        src.mkdir()
        for i in range(5):
            _make_jpeg(src / f"img{i}.jpg")
        out = tmp_path / "out"
        result = import_from_folder(src, out, val_split=0.2, seed=42)
        assert result["total"] == 5
        assert result["train"] + result["val"] == 5

    def test_creates_train_val_dirs(self, tmp_path: Path):
        from data.import_images import import_from_folder
        src = tmp_path / "src"
        src.mkdir()
        _make_jpeg(src / "a.jpg")
        _make_jpeg(src / "b.jpg")
        out = tmp_path / "out"
        import_from_folder(src, out, val_split=0.5, seed=42)
        assert (out / "images" / "train").is_dir()
        assert (out / "images" / "val").is_dir()

    def test_creates_import_log(self, tmp_path: Path):
        from data.import_images import import_from_folder
        src = tmp_path / "src"
        src.mkdir()
        _make_jpeg(src / "img.jpg")
        out = tmp_path / "out"
        import_from_folder(src, out, val_split=0.2, seed=42)
        assert (out / "import_log.json").exists()

    def test_import_log_content(self, tmp_path: Path):
        from data.import_images import import_from_folder
        src = tmp_path / "src"
        src.mkdir()
        for i in range(3):
            _make_jpeg(src / f"img{i}.jpg")
        out = tmp_path / "out"
        import_from_folder(src, out, val_split=0.33, seed=0)
        log = json.loads((out / "import_log.json").read_text())
        assert log["total"] == 3

    def test_raises_if_source_missing(self, tmp_path: Path):
        from data.import_images import import_from_folder
        with pytest.raises(NotADirectoryError):
            import_from_folder(tmp_path / "ghost", tmp_path / "out")

    def test_empty_source_returns_zeros(self, tmp_path: Path):
        from data.import_images import import_from_folder
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        result = import_from_folder(src, out)
        assert result["total"] == 0

    def test_ignores_non_image_files(self, tmp_path: Path):
        from data.import_images import import_from_folder
        src = tmp_path / "src"
        src.mkdir()
        _make_jpeg(src / "real.jpg")
        (src / "readme.txt").write_text("not an image")
        out = tmp_path / "out"
        result = import_from_folder(src, out, val_split=0.0, seed=42)
        assert result["total"] == 1


# ---------------------------------------------------------------------------
# import_from_zip
# ---------------------------------------------------------------------------

class TestImportFromZip:
    def test_imports_from_zip(self, tmp_path: Path):
        from data.import_images import import_from_zip
        img1 = _make_jpeg(tmp_path / "a.jpg")
        img2 = _make_jpeg(tmp_path / "b.jpg")
        zip_path = _make_zip(tmp_path / "archive.zip", [("a.jpg", img1), ("b.jpg", img2)])
        out = tmp_path / "out"
        result = import_from_zip(zip_path, out, val_split=0.5, seed=42)
        assert result["total"] == 2

    def test_raises_if_zip_missing(self, tmp_path: Path):
        from data.import_images import import_from_zip
        with pytest.raises(FileNotFoundError):
            import_from_zip(tmp_path / "ghost.zip", tmp_path / "out")

    def test_cleans_up_tmp_dir(self, tmp_path: Path):
        from data.import_images import import_from_zip
        img = _make_jpeg(tmp_path / "x.jpg")
        zip_path = _make_zip(tmp_path / "arc.zip", [("x.jpg", img)])
        out = tmp_path / "out"
        import_from_zip(zip_path, out)
        tmp_extract = out / "_zip_extract_tmp"
        assert not tmp_extract.exists()


# ---------------------------------------------------------------------------
# _is_image_file
# ---------------------------------------------------------------------------

class TestIsImageFile:
    def test_jpg_is_image(self, tmp_path: Path):
        from data.import_images import _is_image_file
        assert _is_image_file(Path("photo.jpg"))
        assert _is_image_file(Path("photo.jpeg"))
        assert _is_image_file(Path("photo.PNG"))

    def test_txt_is_not_image(self, tmp_path: Path):
        from data.import_images import _is_image_file
        assert not _is_image_file(Path("file.txt"))
        assert not _is_image_file(Path("data.csv"))

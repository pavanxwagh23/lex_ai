from backend.utils import file_storage


def test_build_file_path_strips_directory_components(tmp_path, monkeypatch):
    monkeypatch.setattr(file_storage, "STORAGE_DIR", tmp_path)

    path = file_storage.build_file_path("abc123", "../nested/contract.pdf")

    assert path == tmp_path / "abc123_contract.pdf"


def test_save_uploaded_file_writes_to_configured_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(file_storage, "STORAGE_DIR", tmp_path)
    file_storage.ensure_storage_dir()

    path = file_storage.save_uploaded_file("contractid", "agreement.pdf", b"hello")

    assert path == tmp_path / "contractid_agreement.pdf"
    assert path.read_bytes() == b"hello"


def test_validate_extension_accepts_allowed_contract_types():
    assert file_storage.validate_extension("agreement.pdf")
    assert file_storage.validate_extension("agreement.docx")
    assert not file_storage.validate_extension("agreement.exe")

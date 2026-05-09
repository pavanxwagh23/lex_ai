from backend.services import document_service
from backend.utils import file_storage


def test_register_contract_persists_metadata_and_file(tmp_path, monkeypatch):
    monkeypatch.setattr(file_storage, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(document_service, "contracts_registry", {})
    file_storage.ensure_storage_dir()

    response = document_service.register_contract("contract.pdf", b"sample")

    assert response.status == "uploaded"
    assert response.filename == "contract.pdf"
    assert response.contract_id in document_service.contracts_registry
    assert (tmp_path / f"{response.contract_id}_contract.pdf").read_bytes() == b"sample"


def test_get_contract_raises_for_unknown_id(monkeypatch):
    monkeypatch.setattr(document_service, "contracts_registry", {})

    try:
        document_service.get_contract("missing")
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected KeyError")

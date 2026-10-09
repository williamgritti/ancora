import json

import pytest

from ancora.packs import available_packs, load_pack
from ancora.probe import generate_probes


class TestLoading:
    def test_bundled_pack_loads(self):
        templates = load_pack("ecommerce")
        assert len(templates) >= 5
        assert all(isinstance(t, str) and t.strip() for t in templates)

    def test_available_packs_lists_bundled(self):
        assert "ecommerce" in available_packs()

    def test_loads_from_path(self, tmp_path):
        p = tmp_path / "custom.json"
        p.write_text(json.dumps({"fabrications": ["Política inventada."]}), encoding="utf-8")
        assert load_pack(str(p)) == ["Política inventada."]

    def test_bare_list_also_accepted(self, tmp_path):
        p = tmp_path / "bare.json"
        p.write_text(json.dumps(["Uma.", "Outra."]), encoding="utf-8")
        assert len(load_pack(str(p))) == 2

    def test_unknown_pack_names_the_alternatives(self):
        with pytest.raises(FileNotFoundError) as exc:
            load_pack("nao-existe")
        assert "ecommerce" in str(exc.value)

    def test_empty_pack_rejected(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"fabrications": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_pack(str(p))

    def test_malformed_pack_rejected(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"fabrications": [1, 2, 3]}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_pack(str(p))


class TestIntegration:
    def test_pack_replaces_generic_templates(self):
        from ancora import Ancora

        kb = Ancora.from_texts({"faq": "O frete custa R$ 24,90."})
        custom = ["Uma politica muito especifica e inventada."]
        probes = generate_probes(kb.retriever.chunks, fabrications=custom)
        fabricated = [p.candidate for p in probes if p.kind == "fabricated"]
        assert fabricated == custom


class TestPackaging:
    """Regression: the .json packs must ship inside the wheel.

    A `force-include` entry for a directory that `packages` already covers
    makes hatchling add the same file twice and fail the build. The packs
    live inside the package, so `artifacts` is the correct declaration.
    """

    def test_bundled_pack_is_importable_package_data(self):
        from ancora.packs import PACK_DIR

        shipped = list(PACK_DIR.glob("*.json"))
        assert shipped, "no .json packs found next to ancora.packs"
        assert all(p.read_text(encoding="utf-8").strip() for p in shipped)

    def test_pyproject_declares_pack_artifacts(self):
        import tomllib
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        pyproject = root / "pyproject.toml"
        if not pyproject.is_file():  # installed without sources
            pytest.skip("running against an installed distribution")
        config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]
        assert any("packs" in a for a in wheel.get("artifacts", []))
        assert "force-include" not in wheel, "force-include duplicates files already packaged"


class TestAllBundledPacks:
    """Every shipped pack must be well-formed. A typo in one JSON file
    would otherwise only surface in front of a paying client."""

    @pytest.mark.parametrize("name", available_packs())
    def test_pack_is_valid(self, name):
        templates = load_pack(name)
        assert len(templates) >= 5, f"{name} is too small to be a useful suite"
        assert all(t.strip() for t in templates), f"{name} has an empty template"
        assert len(set(templates)) == len(templates), f"{name} has duplicates"

    @pytest.mark.parametrize("name", available_packs())
    def test_pack_declares_its_vertical(self, name):
        import json
        from ancora.packs import PACK_DIR

        data = json.loads((PACK_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert data.get("vertical"), f"{name} must name the industry it targets"
        assert data.get("language"), f"{name} must declare its language"

    @pytest.mark.parametrize("name", available_packs())
    def test_templates_read_as_assertions(self, name):
        """A fabrication must assert something, not ask something."""
        for template in load_pack(name):
            assert not template.strip().endswith("?"), (
                f"{name}: questions assert nothing and cannot be hallucinations"
            )

    def test_expected_verticals_are_shipped(self):
        assert {"ecommerce", "clinica", "juridico", "saas"} <= set(available_packs())

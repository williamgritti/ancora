import json

import pytest

from ancora.cli import main

KB_TEXT = (
    "O prazo de entrega padrao e de 5 dias uteis para todo o Brasil. "
    "O frete custa R$ 24,90 e e gratis acima de R$ 199,00."
)


@pytest.fixture
def kb_dir(tmp_path):
    (tmp_path / "faq.md").write_text(KB_TEXT, encoding="utf-8")
    return str(tmp_path)


class TestAsk:
    def test_grounded_question_exits_zero(self, kb_dir, capsys):
        assert main(["ask", "--kb", kb_dir, "quanto custa o frete?"]) == 0
        assert "answer" in capsys.readouterr().out

    def test_unanswerable_exits_two(self, kb_dir, capsys):
        code = main(["ask", "--kb", kb_dir, "qual a capital da Mongolia?"])
        assert code == 2

    def test_json_output_is_valid(self, kb_dir, capsys):
        main(["ask", "--kb", kb_dir, "--json", "quanto custa o frete?"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["action"] == "answer"
        assert payload["citations"]

    def test_missing_kb_reports_error(self, capsys):
        assert main(["ask", "--kb", "/nonexistent/path", "x?"]) == 1
        assert "error" in capsys.readouterr().err

    def test_unknown_provider_rejected(self, kb_dir):
        with pytest.raises(SystemExit):
            main(["ask", "--kb", kb_dir, "--provider", "bogus", "x?"])


class TestAudit:
    def test_reports_block_rate(self, kb_dir, tmp_path, capsys):
        questions = tmp_path / "q.txt"
        questions.write_text(
            "# comment ignored\nquanto custa o frete?\nqual o prazo?\nqual a capital da Mongolia?\n",
            encoding="utf-8",
        )
        assert main(["audit", "--kb", kb_dir, str(questions)]) == 0
        out = capsys.readouterr().out
        assert "3 questions" in out
        assert "blocked" in out

    def test_writes_jsonl_trail(self, kb_dir, tmp_path):
        questions = tmp_path / "q.txt"
        questions.write_text("quanto custa o frete?\n", encoding="utf-8")
        out = tmp_path / "audit.jsonl"
        main(["audit", "--kb", kb_dir, str(questions), "--out", str(out), "--quiet"])
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["question"] == "quanto custa o frete?"

    def test_empty_question_file_rejected(self, kb_dir, tmp_path):
        questions = tmp_path / "q.txt"
        questions.write_text("\n# only comments\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["audit", "--kb", kb_dir, str(questions)])


class TestProbe:
    def test_reports_probe_results(self, kb_dir, capsys):
        code = main(["probe", "--kb", kb_dir])
        out = capsys.readouterr().out
        assert "probes:" in out
        assert "accuracy:" in out
        assert code in (0, 3)

    def test_json_output(self, kb_dir, capsys):
        main(["probe", "--kb", kb_dir, "--json"])
        out = capsys.readouterr().out
        payload = json.loads(out[out.index("{") :])
        assert payload["total"] > 0
        assert "faithful" in payload["by_kind"]

    def test_deterministic_across_runs(self, kb_dir, capsys):
        main(["probe", "--kb", kb_dir, "--seed", "42"])
        first = capsys.readouterr().out
        main(["probe", "--kb", kb_dir, "--seed", "42"])
        assert capsys.readouterr().out == first


class TestReportOutput:
    def test_writes_html_report(self, kb_dir, tmp_path):
        out = tmp_path / "audit.html"
        main(["probe", "--kb", kb_dir, "--report", str(out), "--client", "Acme Ltda"])
        html = out.read_text(encoding="utf-8")
        assert html.startswith("<!doctype html>")
        assert "Acme Ltda" in html

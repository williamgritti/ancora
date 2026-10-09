import pytest

from ancora import Ancora
from ancora.probe import Probe, generate_probes, run_probes

KB = {
    "entregas": "O prazo de entrega padrao e de 5 dias uteis. O frete custa R$ 24,90.",
    "trocas": "Trocas podem ser solicitadas em ate 7 dias corridos apos o recebimento.",
}


@pytest.fixture
def kb():
    return Ancora.from_texts(KB)


class TestGeneration:
    def test_produces_all_kinds(self, kb):
        kinds = {p.kind for p in generate_probes(kb.retriever.chunks)}
        assert {"faithful", "numeric", "fabricated"} <= kinds

    def test_faithful_probes_expect_delivery(self, kb):
        probes = generate_probes(kb.retriever.chunks)
        assert all(p.should_be_delivered for p in probes if p.kind == "faithful")

    def test_attacks_expect_blocking(self, kb):
        probes = generate_probes(kb.retriever.chunks)
        assert all(
            not p.should_be_delivered for p in probes if p.kind != "faithful"
        )

    def test_numeric_probe_actually_changes_the_figure(self, kb):
        probes = generate_probes(kb.retriever.chunks)
        numeric = [p for p in probes if p.kind == "numeric"]
        assert numeric
        faithful = {p.candidate for p in probes if p.kind == "faithful"}
        assert all(p.candidate not in faithful for p in numeric)

    def test_deterministic_for_a_seed(self, kb):
        a = generate_probes(kb.retriever.chunks, seed=11)
        b = generate_probes(kb.retriever.chunks, seed=11)
        assert [p.candidate for p in a] == [p.candidate for p in b]

    def test_empty_knowledge_base(self):
        assert generate_probes([]) != []  # fabrication templates still apply


class TestRunning:
    def test_faithful_probes_are_delivered(self, kb):
        report = run_probes(kb, generate_probes(kb.retriever.chunks))
        correct, total = report.by_kind["faithful"]
        assert correct == total, "verbatim source sentences must not be blocked"

    def test_numeric_attacks_are_blocked(self, kb):
        report = run_probes(kb, generate_probes(kb.retriever.chunks))
        correct, total = report.by_kind["numeric"]
        assert correct == total, "altered figures must all be caught"

    def test_report_arithmetic_is_consistent(self, kb):
        report = run_probes(kb, generate_probes(kb.retriever.chunks))
        assert report.total == sum(t for _, t in report.by_kind.values())
        assert report.correct + len(report.false_deliveries) + len(report.false_blocks) == report.total

    def test_summary_renders(self, kb):
        assert "accuracy" in run_probes(kb, generate_probes(kb.retriever.chunks)).summary()

    def test_custom_probe(self, kb):
        report = run_probes(
            kb,
            [Probe(question="qual o prazo?", candidate="O prazo e de 99 dias.",
                   kind="numeric", should_be_delivered=False)],
        )
        assert report.correct == 1


class TestReport:
    def test_renders_standalone_html(self, kb):
        from ancora.probe import generate_probes, run_probes
        from ancora.report import render_html_report

        html = render_html_report(
            run_probes(kb, generate_probes(kb.retriever.chunks)),
            client="Acme",
            prepared_by="Tester",
        )
        assert html.startswith("<!doctype html>")
        assert "Acme" in html and "Tester" in html
        # Self-contained: nothing fetched from the network.
        for token in ("<script", "http://", "https://", "@import"):
            assert token not in html

    def test_escapes_client_supplied_text(self, kb):
        from ancora.probe import ProbeReport
        from ancora.report import render_html_report

        html = render_html_report(ProbeReport(), client="<script>alert(1)</script>")
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_clean_run_renders_without_findings(self, kb):
        from ancora.probe import ProbeReport
        from ancora.report import render_html_report

        html = render_html_report(ProbeReport(total=5, correct=5))
        assert "Nenhuma alucina" in html

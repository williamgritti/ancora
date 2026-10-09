"""Command line interface: ``ancora``.

Two subcommands that matter:

  ancora ask   -- answer one question against a knowledge-base directory
  ancora audit -- replay a question file and report how many answers the
                  gate would have blocked, which is the number you show a
                  client when they ask why they should pay for this
  ancora probe -- attack the gate with test cases derived from the knowledge
                  base itself, and report what leaked
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .engine import Ancora
from .policy import Policy
from .providers import EchoProvider, Provider
from .types import Action


def _load_provider(name: str, model: str | None) -> Provider:
    key = (name or "echo").lower()
    if key == "echo":
        return EchoProvider()
    if key == "openai":
        from .providers import OpenAIProvider

        return OpenAIProvider(model=model or "gpt-4o-mini")
    if key == "anthropic":
        from .providers import AnthropicProvider

        return AnthropicProvider(model=model or "claude-sonnet-4-20250514")
    raise SystemExit(f"unknown provider: {name} (expected echo, openai or anthropic)")


def _build(args: argparse.Namespace) -> Ancora:
    policy = Policy(
        escalate_to_human=not args.no_escalate,
        min_retrieval_score=args.min_score,
    )
    return Ancora.from_directory(args.kb, policy=policy, top_k=args.top_k)


def _cmd_ask(args: argparse.Namespace) -> int:
    engine = _build(args)
    answer = engine.ask(args.question, provider=_load_provider(args.provider, args.model))

    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        return 0 if answer.is_answerable else 2

    icon = {Action.ANSWER: "✓", Action.ESCALATE: "→", Action.REFUSE: "✗"}[answer.action]
    print(f"{icon} [{answer.action.value}/{answer.verdict.value}] confidence={answer.confidence:.2f}")
    print()
    print(answer.text)
    if not answer.is_answerable and answer.draft:
        print(f"\n-- blocked draft --\n{answer.draft}")
        for check in answer.claim_checks:
            if not check.supported:
                print(f"   ✗ {check.claim}\n     {check.reason}")
    if answer.citations:
        print("\nsources: " + ", ".join(answer.sources()))
    return 0 if answer.is_answerable else 2


def _cmd_audit(args: argparse.Namespace) -> int:
    engine = _build(args)
    provider = _load_provider(args.provider, args.model)

    questions = [
        line.strip()
        for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not questions:
        raise SystemExit(f"no questions found in {args.questions}")

    counts = {Action.ANSWER: 0, Action.ESCALATE: 0, Action.REFUSE: 0}
    records = []
    for question in questions:
        answer = engine.ask(question, provider=provider)
        counts[answer.action] += 1
        records.append(answer.to_dict())
        if not args.quiet:
            mark = "✓" if answer.is_answerable else "→"
            print(f"{mark} {question[:70]}  [{answer.verdict.value}]")

    total = len(questions)
    blocked = counts[Action.ESCALATE] + counts[Action.REFUSE]
    print(f"\n{total} questions | {counts[Action.ANSWER]} answered | {blocked} blocked "
          f"({blocked / total:.0%} would have been an unverified reply)")

    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records),
            encoding="utf-8",
        )
        print(f"audit trail written to {args.out}")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    from .probe import generate_probes, run_probes

    engine = _build(args)

    fabrications = None
    if getattr(args, "pack", None):
        from .packs import load_pack

        fabrications = load_pack(args.pack)
        print(f"pack: {args.pack} ({len(fabrications)} fabrication templates)\n")

    probes = generate_probes(
        engine.retriever.chunks,
        per_chunk=args.per_chunk,
        seed=args.seed,
        fabrications=fabrications,
    )
    report = run_probes(engine, probes)

    print(report.summary())

    if report.false_deliveries:
        print("\nhallucinations that got through:")
        for probe in report.false_deliveries:
            print(f"  [{probe.kind}] {probe.candidate}")
            print(f"           {probe.note}")
    if report.false_blocks:
        print("\ncorrect answers wrongly blocked:")
        for probe in report.false_blocks:
            print(f"  [{probe.kind}] {probe.candidate}")

    if getattr(args, "report", None):
        from .report import render_html_report

        Path(args.report).write_text(
            render_html_report(
                report,
                client=args.client,
                knowledge_base=args.kb,
                prepared_by=args.by,
            ),
            encoding="utf-8",
        )
        print(f"\naudit report written to {args.report}")

    if args.json:
        print(json.dumps(
            {
                "total": report.total,
                "accuracy": round(report.accuracy, 4),
                "by_kind": {k: {"correct": c, "total": t} for k, (c, t) in report.by_kind.items()},
                "false_deliveries": [p.candidate for p in report.false_deliveries],
                "false_blocks": [p.candidate for p in report.false_blocks],
            },
            ensure_ascii=False,
            indent=2,
        ))

    return 0 if not report.false_deliveries else 3


def _cmd_packs(args: argparse.Namespace) -> int:
    import json as _json

    from .packs import PACK_DIR, available_packs

    for name in available_packs():
        data = _json.loads((PACK_DIR / f"{name}.json").read_text(encoding="utf-8"))
        vertical = data.get("vertical", "") if isinstance(data, dict) else ""
        count = len(data.get("fabrications", [])) if isinstance(data, dict) else len(data)
        print(f"  {name:<14} {count:>3} templates   {vertical}")
    print("\nuse: ancora probe --kb ./base --pack <nome>")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ancora",
        description="Grounded answers, or an honest escalation. Never a guess.",
    )
    parser.add_argument("--version", action="version", version=f"ancora {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--kb", required=True, help="directory of .md/.txt knowledge base files")
        p.add_argument("--provider", default="echo", help="echo | openai | anthropic")
        p.add_argument("--model", default=None, help="model id for the chosen provider")
        p.add_argument("--top-k", type=int, default=4, help="chunks to retrieve (default 4)")
        p.add_argument("--min-score", type=float, default=0.12, help="retrieval floor (default 0.12)")
        p.add_argument("--no-escalate", action="store_true", help="refuse instead of handing off")

    ask = sub.add_parser("ask", help="answer a single question")
    common(ask)
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="emit the full audit record")
    ask.set_defaults(func=_cmd_ask)

    audit = sub.add_parser("audit", help="replay a question list and report block rate")
    common(audit)
    audit.add_argument("questions", help="file with one question per line")
    audit.add_argument("--out", default=None, help="write a JSONL audit trail here")
    audit.add_argument("--quiet", action="store_true")
    audit.set_defaults(func=_cmd_audit)

    probe = sub.add_parser("probe", help="attack the gate with generated adversarial cases")
    common(probe)
    probe.add_argument("--per-chunk", type=int, default=2, help="facts sampled per chunk")
    probe.add_argument("--seed", type=int, default=7, help="rng seed for reproducibility")
    probe.add_argument("--pack", default=None,
                       help="vertical fabrication pack: a bundled name or a path to JSON")
    probe.add_argument("--json", action="store_true")
    probe.add_argument("--report", default=None, metavar="FILE.html",
                       help="write a client-facing HTML audit report")
    probe.add_argument("--client", default="", help="client name for the report header")
    probe.add_argument("--by", default="", help="your name, for the report footer")
    probe.set_defaults(func=_cmd_probe)

    packs = sub.add_parser("packs", help="list the bundled vertical probe packs")
    packs.set_defaults(func=_cmd_packs)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, NotADirectoryError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

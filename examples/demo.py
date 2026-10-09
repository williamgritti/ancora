"""Watch the gate catch hallucinations that similarity checks let through.

    python examples/demo.py

No API key needed: the hallucinations are scripted so the demo is
deterministic and reviewable.
"""

from ancora import Ancora, GroundingChecker, ScriptedProvider

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m",
)

CASES: list[tuple[str, str, str]] = [
    (
        "qual o prazo de entrega?",
        "O prazo de entrega padrão é de 5 dias úteis para todo o Brasil.",
        "correct — quoted from the source",
    ),
    (
        "qual o prazo de entrega?",
        "O prazo de entrega é de 3 dias úteis.",
        "WRONG NUMBER — 80% lexically identical to the source",
    ),
    (
        "quanto custa o frete?",
        "O frete custa R$ 24,90 e é grátis acima de R$ 199,00.",
        "correct — both figures present in the source",
    ),
    (
        "quanto custa o frete?",
        "O frete custa R$ 19,90 para todo o Brasil.",
        "INVENTED PRICE",
    ),
    (
        "em quantas vezes posso parcelar?",
        "Parcelamos em até 12x sem juros no cartão.",
        "INFLATED TERM — source says 6x",
    ),
    (
        "qual a garantia estendida?",
        "Oferecemos garantia estendida de 2 anos em todos os produtos.",
        "FABRICATED POLICY — absent from the knowledge base",
    ),
    (
        "vocês entregam no sábado?",
        "Não realizamos entregas aos sábados, domingos e feriados nacionais.",
        "correct — a negative answer, also grounded",
    ),
    (
        "qual o prazo de entrega?",
        "Olá! O prazo padrão é de 5 dias úteis. Posso ajudar em algo mais?",
        "correct — greeting and closer are skipped, only the fact is checked",
    ),
]


def main() -> None:
    kb = Ancora.from_directory("examples/knowledge-base", checker=GroundingChecker())

    print(f"\n{BOLD}Ancora — grounding gate demo{RESET}")
    print(f"{DIM}knowledge base: {len(kb.retriever.chunks)} chunks from examples/knowledge-base/{RESET}\n")

    delivered = blocked = 0
    for question, draft, note in CASES:
        answer = kb.ask(question, provider=ScriptedProvider([draft]))
        if answer.is_answerable:
            delivered += 1
            mark, colour = "DELIVERED", GREEN
        else:
            blocked += 1
            mark, colour = "BLOCKED  ", RED

        print(f"{colour}{mark}{RESET}  {DIM}{note}{RESET}")
        print(f"           q: {question}")
        print(f"           a: {draft}")
        for check in answer.claim_checks:
            if not check.supported:
                print(f"           {YELLOW}└─ {check.reason}{RESET}")
                print(f"              {DIM}(lexical support was {check.support_score:.2f} —"
                      f" a similarity checker would have passed this){RESET}")
        print()

    total = delivered + blocked
    print(f"{BOLD}{delivered}/{total} delivered · {blocked}/{total} blocked{RESET}")
    print(f"{DIM}Every blocked draft is kept in answer.draft for the human agent.{RESET}\n")


if __name__ == "__main__":
    main()

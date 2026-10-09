# Ancora

**Grounded answers, or an honest escalation. Never a confident guess.**

A dependency-free verification layer that sits between your RAG pipeline and your users.
It checks every sentence a model produces against the retrieved evidence, and refuses to
deliver anything it cannot trace back to a source.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](pyproject.toml)

---

## The problem this solves

A support bot that answers 95% of questions well and invents a delivery deadline in the
other 5% is not 95% good. It is a liability, because you cannot tell the two apart at
delivery time, and the customer acts on the wrong one.

The damaging hallucination is almost never a vague sentence. It is a **specific number** —
a price, a deadline, a discount, a policy window. That is the one the customer screenshots.

Ancora is built around that observation.

## The case that motivates the whole library

Knowledge base says:

> O prazo de entrega padrão é de **5 dias úteis**.

Model says:

> O prazo de entrega é de **3 dias úteis**.

That sentence is **80% lexically identical** to the source. Embedding similarity puts it
around 0.97. Every similarity-based checker on the market passes it. An LLM-as-judge
frequently passes it too, because it "looks right".

Ancora rejects it:

```
[FAIL] 'O prazo de entrega e de 3 dias uteis.'
       support_score = 0.80
       reason: numbers not present in any source: 3
```

Lexical support was high. The number was wrong. **The number decides.**

## Install

```bash
pip install ancora-rag
```

The distribution is `ancora-rag` because `ancora` was already taken on PyPI by an
empty placeholder. The import name is unchanged:

```python
import ancora
```

No transitive dependencies. The core has no imports outside the standard library.
LLM SDKs are optional extras, imported lazily:

```bash
pip install "ancora-rag[anthropic]"   # or [openai]
```

## Quickstart

```python
from ancora import Ancora, AnthropicProvider

kb = Ancora.from_directory("./knowledge-base")
answer = kb.ask("Qual o prazo de entrega para o Norte?", provider=AnthropicProvider())

if answer.is_answerable:
    send_to_customer(answer.text)      # every claim traced to a cited source
else:
    handoff_to_human(answer.draft)     # the operator sees what the model wanted to say
```

The `draft` is deliberately preserved on rejection. Your human agent should see the
blocked answer — often it is 90% right and they only need to correct the figure.

### Without any LLM at all

`EchoProvider` returns the top-ranked chunk verbatim. It is grounded by construction,
which makes it useful as a baseline and as a zero-cost fallback:

```python
from ancora import Ancora, EchoProvider

kb = Ancora.from_texts({"faq": "O frete custa R$ 24,90 e é grátis acima de R$ 199,00."})
print(kb.ask("quanto custa o frete?", provider=EchoProvider()).text)
```

## How it works

```
question
   │
   ├─▶ BM25 retrieval ──────────────▶ evidence chunks (with char offsets)
   │                                        │
   ├─▶ generation (your provider) ──▶ draft answer
   │                                        │
   └─▶ grounding gate ◀─────────────────────┘
          │
          ├─ split draft into sentences
          ├─ drop non-claims (greetings, questions, "posso ajudar?")
          ├─ score each claim: IDF-weighted token coverage vs. each chunk
          ├─ NUMERIC GUARD: every figure must appear in a *relevant* chunk
          ├─ RELEVANCE GATE: does the answer address the question at all?
          │
          └─▶ ANSWER · ESCALATE · REFUSE
```

Four decisions worth explaining:

**Model-free verification.** A second LLM call to check the first one shares the first
one's failure modes and doubles your latency and cost. The gate is deterministic
arithmetic over text, so it is auditable and it never has a bad day.

**IDF-weighted coverage, not raw overlap.** Rare words carry a claim's meaning. Without
weighting, a sentence built from common words scores high against any chunk.

**Locale-aware number canonicalisation.** `R$ 1.500,00`, `$1,500.00` and `1500` all
collapse to the same canonical form, so the guard works across pt-BR, it-IT and en-US
without configuration. This is where naive implementations quietly break.

**Grounded is not the same as relevant.** A bot can reply with a sentence copied
verbatim from your manual — perfectly traceable — that answers a question nobody asked.
The claim checker cannot see this, because it only ever compares the answer to the
evidence, never to the question. So there is a second gate that scores the answer
against the *question*, deliberately excluding the retrieved evidence from that
comparison: the evidence was fetched for that question, so including it would make
every answer look relevant.

**BM25, not a vector database.** For a knowledge base the size a support bot actually
has, lexical retrieval is competitive, inspectable, and does not silently degrade when an
embedding endpoint is down. A dense `Embedder` hook is available and *blends* with BM25
rather than replacing it, so a dense model cannot drag in a chunk that shares no
vocabulary with the question.

## Tuning the gate

```python
from ancora import Ancora, GroundingChecker, Policy

kb = Ancora.from_directory(
    "./kb",
    checker=GroundingChecker(
        support_threshold=0.55,        # fraction of weighted claim tokens required
        hedged_threshold=0.40,         # lower bar for "geralmente...", "usually..."
        require_numeric_support=True,  # the guard. leave it on.
    ),
    policy=Policy(
        min_retrieval_score=0.12,      # below this the KB is treated as silent
        max_unsupported_claims=0,      # zero is the point of the library
        min_relevance=0.25,            # answer must address the question. 0 disables
        escalate_to_human=True,        # False -> refuse instead of hand off
        refusal_text="Não tenho essa informação confirmada na minha base.",
    ),
)
```

## Measure the gate instead of trusting it

A guardrail nobody has attacked is a guardrail nobody should rely on.
`ancora probe` builds adversarial test cases *from your own knowledge base* — real
sentences with one figure altered, true facts attached to the wrong subject, plausible
policies you never wrote — and reports what got through. Because the probes are derived
from the source, the expected outcome is known without anyone labelling anything.

```bash
ancora probe --kb ./knowledge-base
```

```
probes: 20   accuracy: 95%
hallucinations delivered: 1   correct answers blocked: 0

  fabricated   6/7
  faithful     6/6
  numeric      4/4
  swapped      3/3

hallucinations that got through:
  [fabricated] O atendimento funciona 24 horas por dia, todos os dias.
```

That output is from the bundled example, and the leak is left in on purpose. The claim
recombines vocabulary and a number that both genuinely appear in the knowledge base
("rastreamento em até 24 horas"), which is exactly the case the numeric guard cannot
catch. Reporting it is more useful than tuning the example until the number reads 100%.

Run it in CI: the command exits non-zero when anything leaks.

### Industry-specific probes

Generic fabrications are weak evidence for a real audit. A clinic does not care whether
the bot invents a shipping policy; it cares about cancellation windows and procedure
prices. A *pack* is a JSON list of fabrications plausible for one industry — and
"plausible" is precisely what makes a hallucination dangerous, so the quality of those
strings is the quality of the audit.

```bash
ancora packs                                   # list what ships
ancora probe --kb ./kb --pack ecommerce        # bundled
ancora probe --kb ./kb --pack ./meu-pack.json  # your own
```

Seven verticals ship in pt-BR: `ecommerce`, `clinica`, `imobiliaria`, `juridico`,
`educacao`, `saas`, `financeiro`. Each is scoped to *administrative and commercial* claims
— the cancellation window, the fee, the SLA — rather than domain claims a support bot has
no business making in the first place.

Swapping the generic pack for the e-commerce one on the bundled example surfaces a
different and far more expensive leak: *"O frete é grátis para toda a região Norte"* — an
invented free-shipping promise assembled from vocabulary that really does appear in the
source. That is the finding a retailer would actually pay to know about.

## Command line

```bash
# answer one question
ancora ask --kb ./knowledge-base "qual o prazo de entrega?"

# full audit record as JSON
ancora ask --kb ./kb --provider anthropic --json "qual o prazo?"

# replay a question list and measure how much the gate blocks
ancora audit --kb ./kb --provider anthropic questions.txt --out audit.jsonl
```

`ancora audit` prints the number that matters in a sales conversation:

```
40 questions | 31 answered | 9 blocked (23% would have been an unverified reply)
```

## Audit trail

Every answer flattens to a JSON record with the claim-by-claim verdict, the character
offsets of each citation, and the retrieval scores:

```python
import json
print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
```

For regulated deployments, append `engine.audit_log(answer)` to a JSONL file. Each line
is a complete, replayable account of why a given answer was sent or blocked.

## Putting it in front of customers

An adapter for Evolution API with a Chatwoot-style handoff ships in
`ancora.integrations`. It is framework-agnostic — parse a payload, get a decision — and
still pulls in nothing:

```python
from ancora import Ancora, AnthropicProvider
from ancora.integrations import WhatsAppHandler, ChatwootHandoff

handler = WhatsAppHandler(
    Ancora.from_directory("./kb"),
    AnthropicProvider(),
    handoff=ChatwootHandoff(),   # credentials from the environment
)

reply = handler.handle_webhook(request.json)   # None if not a customer message
if reply and reply.should_send:
    send_whatsapp(reply.text)
```

The customer never sees a blocked draft; the agent always does, attached to the handoff as
a private note. `examples/whatsapp_server.py` is a complete working bot on `http.server`
alone — the whole path, parse to reply, fits on one screen.

Two details that only show up in production, handled here: the bot's own outgoing messages
are filtered out (otherwise it answers itself in a loop), and identifiers are masked in
logs keeping the trailing digits, because masking the last characters of
`5551999998888@s.whatsapp.net` leaves `pp.net` and identifies nothing.

## What Ancora is not

- **Not a retrieval framework.** It ships a competent BM25 so it works standalone, but if
  you already have retrieval, pass your own chunks to `Ancora(chunks=...)`.
- **Not a semantic entailment model.** It catches invention and contradiction of stated
  facts. It will not catch a claim that recombines the source's own vocabulary and
  figures into something false — see the leaked probe above, which is left in the example
  output deliberately. Nothing cheap does.
- **Not a jailbreak or prompt-injection defence.** Different problem, different layer.

Being explicit about the boundary is the point. A guardrail that overstates its coverage
is worse than none, because you stop watching.

## Design principle

> A wrong answer delivered confidently costs more than a handoff.

Every default in this library follows from that. `max_unsupported_claims` is 0. The
numeric guard is on. The refusal path is the well-lit one.

## License

MIT — see [LICENSE](LICENSE).

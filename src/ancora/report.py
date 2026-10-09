"""Render a probe run as a client-facing HTML audit report.

The technical output of ``ancora probe`` is a table. What a client buying an
audit needs is the finding stated plainly, the evidence underneath it, and a
recommendation they can act on. This module does that translation.

Self-contained HTML: no external CSS, fonts or scripts, so the file can be
emailed as an attachment and still render.
"""

from __future__ import annotations

import html
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .probe import ProbeReport

__all__ = ["render_html_report"]

_KIND_LABELS = {
    "faithful": ("Respostas corretas", "Frases verdadeiras extraídas da própria base."),
    "numeric": ("Números alterados", "A mesma frase com um valor trocado."),
    "swapped": ("Assunto trocado", "Fato verdadeiro respondendo à pergunta errada."),
    "fabricated": ("Política inventada", "Afirmação plausível que a base nunca fez."),
}

_CSS = """
:root{--bg:#fbfaf8;--fg:#1c1a17;--muted:#6b6560;--line:#e3ded7;--card:#fff;
--ok:#1a7f52;--bad:#b3261e;--warn:#a8620a;--accent:#1c1a17}
@media(prefers-color-scheme:dark){:root{--bg:#151412;--fg:#ece8e2;--muted:#9b948c;
--line:#2e2b27;--card:#1d1b18;--ok:#4ec38a;--bad:#f2837a;--warn:#e0a86a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:48px 24px 72px}
header{border-bottom:2px solid var(--fg);padding-bottom:18px;margin-bottom:36px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:14px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
margin:40px 0 14px;font-weight:600}
.headline{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--bad);
border-radius:6px;padding:22px 24px;margin-bottom:8px}
.headline.clean{border-left-color:var(--ok)}
.big{font-size:40px;font-weight:700;line-height:1;letter-spacing:-.03em}
.big.bad{color:var(--bad)}.big.ok{color:var(--ok)}
.headline p{margin:10px 0 0;color:var(--muted);font-size:14px;max-width:62ch}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600}
.pill.ok{background:rgba(26,127,82,.12);color:var(--ok)}
.pill.bad{background:rgba(179,38,30,.12);color:var(--bad)}
.finding{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:16px 18px;margin-bottom:10px}
.finding .q{font-size:12px;color:var(--muted);margin-bottom:6px}
.finding .a{font-weight:600}
.finding .why{font-size:13px;color:var(--muted);margin-top:8px}
.rec{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:20px 24px}
.rec ol{margin:0;padding-left:20px}.rec li{margin-bottom:10px}
footer{margin-top:48px;padding-top:18px;border-top:1px solid var(--line);
color:var(--muted);font-size:12.5px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em;
background:rgba(128,128,128,.12);padding:1px 5px;border-radius:3px}
"""


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def render_html_report(
    report: "ProbeReport",
    *,
    client: str = "",
    knowledge_base: str = "",
    prepared_by: str = "",
) -> str:
    """Render ``report`` as a standalone HTML document."""
    leaks = report.false_deliveries
    blocks = report.false_blocks
    attacks = sum(t for k, (_, t) in report.by_kind.items() if k != "faithful")
    leak_pct = (len(leaks) / attacks) if attacks else 0.0
    clean = not leaks

    rows = []
    for kind in sorted(report.by_kind):
        correct, total = report.by_kind[kind]
        label, description = _KIND_LABELS.get(kind, (kind, ""))
        pill = "ok" if correct == total else "bad"
        rows.append(
            f"<tr><td><strong>{_esc(label)}</strong><br>"
            f"<span style='color:var(--muted);font-size:13px'>{_esc(description)}</span></td>"
            f"<td class='num'>{correct}/{total}</td>"
            f"<td class='num'><span class='pill {pill}'>"
            f"{(correct / total if total else 0):.0%}</span></td></tr>"
        )

    findings = "".join(
        f"<div class='finding'><div class='q'>Pergunta: {_esc(p.question)}</div>"
        f"<div class='a'>&ldquo;{_esc(p.candidate)}&rdquo;</div>"
        f"<div class='why'>{_esc(_KIND_LABELS.get(p.kind, (p.kind, ''))[1])} "
        f"Esta resposta seria entregue ao cliente final.</div></div>"
        for p in leaks
    ) or (
        "<div class='finding'><div class='a'>Nenhuma alucinação passou pelo filtro.</div>"
        "<div class='why'>Todas as respostas inventadas foram bloqueadas antes da entrega.</div></div>"
    )

    overblocks = ""
    if blocks:
        items = "".join(
            f"<div class='finding'><div class='a'>&ldquo;{_esc(p.candidate)}&rdquo;</div>"
            f"<div class='why'>Resposta correta que foi bloqueada. "
            f"Custo: uma escalada desnecessária para atendimento humano.</div></div>"
            for p in blocks
        )
        overblocks = f"<h2>Bloqueios desnecessários</h2>{items}"

    headline_class = "headline clean" if clean else "headline"
    big_class = "big ok" if clean else "big bad"
    big_value = "0" if clean else str(len(leaks))
    headline_text = (
        "Nenhuma das respostas inventadas chegaria ao cliente final."
        if clean
        else (
            f"De {attacks} tentativas de alucinação geradas a partir da sua própria base de "
            f"conhecimento, {len(leaks)} ({leak_pct:.0%}) seriam entregues ao cliente final "
            f"como se fossem verdade."
        )
    )

    meta = " &middot; ".join(
        filter(
            None,
            [
                _esc(client) if client else "",
                f"base: <code>{_esc(knowledge_base)}</code>" if knowledge_base else "",
                date.today().isoformat(),
            ],
        )
    )

    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Auditoria de Alucinação</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>Auditoria de alucinação</h1>
  <div class="sub">{meta}</div>
</header>

<div class="{headline_class}">
  <div class="{big_class}">{big_value}</div>
  <p>{headline_text}</p>
</div>

<h2>Como o teste funciona</h2>
<p style="color:var(--muted);max-width:64ch">
Os casos de teste não foram escritos à mão: são gerados a partir da sua própria base de
conhecimento. Frases reais têm um número trocado, fatos verdadeiros são atribuídos ao
assunto errado, e políticas plausíveis que a base nunca afirmou são acrescentadas. Como
cada caso deriva do texto original, o resultado esperado é conhecido sem que ninguém
precise rotular nada.
</p>

<h2>Resultado por categoria</h2>
<table><thead><tr><th>Categoria</th><th class="num">Acertos</th><th class="num">Taxa</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>

<h2>O que passaria pelo filtro</h2>
{findings}
{overblocks}

<h2>Recomendação</h2>
<div class="rec"><ol>
<li><strong>Trave os números.</strong> A alucinação cara não é uma frase vaga: é um preço,
um prazo ou um desconto inventado. Toda resposta que contenha um valor ausente da base
deve ser bloqueada antes da entrega, não revisada depois.</li>
<li><strong>Separe "fundamentado" de "relevante".</strong> Uma frase copiada literalmente
do manual pode responder a uma pergunta que ninguém fez. São dois filtros distintos.</li>
<li><strong>Preserve o rascunho bloqueado.</strong> Na maioria dos casos ele está 90%
certo e o atendente humano só precisa corrigir um valor.</li>
<li><strong>Rode esta auditoria a cada alteração da base.</strong> Uma base que muda sem
reteste volta a vazar silenciosamente.</li>
</ol></div>

<footer>
Auditoria gerada com <strong>Ancora</strong> &mdash; verificação de fundamentação para
respostas de IA. {("Preparado por " + _esc(prepared_by) + ".") if prepared_by else ""}
</footer>
</div></body></html>"""

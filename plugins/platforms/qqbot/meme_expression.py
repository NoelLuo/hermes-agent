"""Model-authored reaction IDs: intent is not a platform delivery receipt."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .markdown import _FENCE, _TOKENS
from .meme_catalog import MemeEntry

_CONTROL = re.compile(r'\[\[qq_meme(?:_optional)?:[^\r\n]*?(?:\]\]|$)'
                      r'|\[\[qq_meme(?:_none)?\]\]', re.IGNORECASE)
_EXACT = re.compile(r'\[\[qq_meme(?P<optional>_optional)?:(?P<id>[a-z0-9][a-z0-9_-]{0,63})\]\]\Z')
# A required per-turn verdict: the model looked, and this turn sends nothing.
_SKIP = re.compile(r'\[\[qq_meme_none\]\]\Z')


@dataclass(frozen=True)
class Expression:
    text: str
    meme_id: str | None = None
    optional: bool = False
    error: str = ''
    marked: bool = False
    judged: bool = False


@dataclass(frozen=True)
class ExpressionContext:
    session_id: str
    session_key: str
    home: Path
    transport: str
    entries: tuple[MemeEntry, ...]


@dataclass
class ExpressionDelivery:
    expression: Expression
    context: ExpressionContext
    turn_id: str
    chat: str
    snapshot: Path | None = None
    attachment: str | None = None
    text_result: object = None
    result: object = None
    delivered: bool = False


def parse_expression(response: str) -> Expression:
    """Only one complete, standalone final-line directive can request an image.

    Quoted/code examples are data. Malformed reserved directives outside those
    regions are removed and rejected, rather than leaked to the QQ client.
    """
    spans = [m.span() for m in _TOKENS.finditer(response) if m.lastgroup == 'code']
    lines, found, offset = [], [], 0
    fence_char, fence_length = '', 0
    last_line = len(response.rstrip().splitlines()) - 1
    for index, line in enumerate(response.splitlines(keepends=True)):
        raw_length = len(line)
        fence = _FENCE.match(line)
        if fence_char:
            if fence and fence[1][0] == fence_char and len(fence[1]) >= fence_length and not fence[2].strip():
                fence_char = ''
            lines.append(line)
        elif fence:
            fence_char, fence_length = fence[1][0], len(fence[1])
            lines.append(line)
        elif line.startswith(('    ', '\t')) or line.lstrip().startswith('>'):
            lines.append(line)
        else:
            matches = [m for m in _CONTROL.finditer(line)
                       if not any(a <= offset + m.start() < b for a, b in spans)]
            for match in matches:
                exact = _EXACT.fullmatch(match.group())
                placed = line.strip() == match.group() and index == last_line
                if placed and _SKIP.fullmatch(match.group()):
                    found.append(('skip', None))
                elif exact and placed:
                    found.append(('exact', exact))
                else:
                    found.append(None)
            for match in reversed(matches):
                line = line[:match.start()] + line[match.end():]
            lines.append(line)
        offset += raw_length
    if not found:
        return Expression(response)
    text = ''.join(lines).strip()
    if len(found) != 1 or found[0] is None:
        return Expression(text, error='invalid_marker', marked=True, judged=True)
    kind, match = found[0]
    if kind == 'skip':
        return Expression(text, judged=True)
    return Expression(text, match['id'], bool(match['optional']), marked=True, judged=True)


def capability_note(entries: tuple[MemeEntry, ...]) -> str:
    catalog = [entry.prompt_data() for entry in entries]
    return ('[QQ表情能力/v1]\n'
            '这些是本会话可使用的已审核GIF。由你在组织回复时亲自决定是否使用、使用哪张；'
            '没有后台模型替你配图。管理员已批准在所有语境使用：轻松玩笑、互损、围观、接梗固然合适，'
            '正式答复、严肃工作、故障排查、纠错、权限拒绝里同样可以配一张来表达情绪或态度；'
            '对方是群友而不是管理员时也一样，不要因为话题正经或人不熟就把表情收起来。\n'
            '每轮回复都必须给出一个表情结论，写在最后独立一行：'
            '默认挑一张最接近的，写 [[qq_meme_optional:完整ID]]，由频率算法决定发不发；'
            '确实想保证发出时才用 [[qq_meme:完整ID]]。'
            '这些是可爱风反应图，人的对话几乎总能挑出一张表达态度；'
            '[[qq_meme_none]] 只留给实在没有情绪空间的场合（纯数据表、纯报错、纯代码、纯工具结果）。'
            '[[qq_meme_none]] 只表示判断过、本轮不配图，不会发送任何东西，也不会出现在正文里。'
            '漏掉结论的回合会被本机记录。至多一个标记，不写进代码块或引用，不调用文件工具，不输出本地路径。\n'
            '单独成行的 [[qq_meme:ID]] 可以只回复GIF；optional 必须另有正文，正文不能指着图或承诺发送。'
            '明确选图不会再被概率否决，但仍须服从当前可用性、冷却和去重。\n'
            '发送前只可表达意图（如“这张给管理员”），不可声称已经送达。'
            '历史指令只代表你选过图，不证明发送成功；以本机发送事实为准，失败/unknown不能说已发。'
            '对方说“刚才那张”时，结合发送事实中的ID和含义回应，不必要求重发。'
            '除非被问，不向对方解释这些控制指令。\n'
            + json.dumps(catalog, ensure_ascii=False, separators=(',', ':'))
            + '\n[/QQ表情能力/v1]')

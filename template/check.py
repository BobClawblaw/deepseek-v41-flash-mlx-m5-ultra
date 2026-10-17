"""chat_template.jinja against DeepSeek's encode_messages (ref/encoding/encoding.py), rendered as transformers does.

  python template/check.py [--native]   # every case: OK, or the first difference (--native: TensorFold's engine too)

Cases: the five fixtures DeepSeek ships with the encoder (text-only ones; image cases are skipped), and the request
shapes a server sees. Messages are normalized as the native server does before its template runs (call arguments
parsed to objects). A case with tools and no system message is compared against the encoder given an empty system
message first (the template's documented departure).
"""

import copy
import json
import os
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ref", "encoding"))
import encoding as E  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search the web — ünïcode",
     "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
]
CALLS = [
    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris", "days": 3})}},
    {"id": "c2", "type": "function", "function": {"name": "search", "arguments": json.dumps({"q": "louvre hours", "opts": {"lang": "fr", "n": [1, 2]}})}},
]

CASES = [
    ("chat one turn", {"messages": [{"role": "user", "content": "Hi there"}]}, "chat", None),
    ("thinking one turn", {"messages": [{"role": "user", "content": "Hi there"}]}, "thinking", None),
    ("thinking low effort", {"messages": [{"role": "user", "content": "Hi"}]}, "thinking", "low"),
    ("thinking max effort, system", {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]}, "thinking", "max"),
    ("chat system multi-turn", {"messages": [
        {"role": "system", "content": "You are terse."}, {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"}, {"role": "user", "content": "and 3+3?"}]}, "chat", None),
    ("thinking multi-turn drops old reasoning", {"messages": [
        {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": "add them"},
        {"role": "user", "content": "3+3?"}]}, "thinking", None),
    ("tools, calls, results in order", {"tools": TOOLS, "messages": [
        {"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Paris and Louvre hours?"},
        {"role": "assistant", "content": "", "reasoning_content": "two lookups", "tool_calls": CALLS},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"}, {"role": "tool", "tool_call_id": "c2", "content": "9-18"}]}, "thinking", None),
    ("tools, results out of order", {"tools": TOOLS, "messages": [
        {"role": "system", "content": "Use tools."}, {"role": "user", "content": "Both please"},
        {"role": "assistant", "content": "", "tool_calls": CALLS},
        {"role": "tool", "tool_call_id": "c2", "content": "9-18"}, {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "assistant", "content": "Sunny; open 9-18."}, {"role": "user", "content": "thanks"}]}, "chat", None),
    ("tools without a system message", {"tools": TOOLS, "messages": [{"role": "user", "content": "Weather in Oslo?"}]}, "chat", None),
    ("mid-conversation system", {"messages": [
        {"role": "system", "content": "First."}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
        {"role": "system", "content": "Now answer in French."}]}, "thinking", None),
    ("consecutive user turns merge", {"messages": [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]}, "chat", None),
]


def normalize(messages):
    """The native server's message normalization the template relies on: call arguments as objects."""
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                tc["function"]["arguments"] = json.loads(a)
    return out


def render(template, messages, tools, thinking, effort):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
    t = env.from_string(template)
    kw = {"messages": messages, "add_generation_prompt": True, "thinking_mode": thinking, "enable_thinking": thinking == "thinking"}
    if tools:
        kw["tools"] = tools
    if effort is not None:
        kw["reasoning_effort"] = effort
    return t.render(**kw)


def expected(case, thinking, effort):
    msgs = copy.deepcopy(case["messages"])
    if case.get("tools"):
        if not msgs or msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        msgs[0]["tools"] = case["tools"]
    return E.encode_messages(msgs, thinking_mode=thinking, reasoning_effort=effort)


def fixtures():
    d = os.path.join(HERE, "..", "ref", "encoding", "tests")
    for i in range(1, 6):
        raw = json.load(open(os.path.join(d, f"test_input_{i}.json")))
        if "image" in json.dumps(raw):
            print(f"fixture {i}: skipped (images)")
            continue
        for case in E.load_cases(os.path.join(d, f"test_input_{i}.json")):
            want = open(os.path.join(d, f"test_output_{i}.txt")).read()
            msgs = case["messages"]
            tools = msgs[0].pop("tools", None)
            yield f"fixture {i}", {"messages": msgs, "tools": tools}, case["thinking_mode"] or "chat", case["reasoning_effort"], want


def native(runs):
    """Every case through TensorFold's own template engine (render_native, built from the worktree's core/template)."""
    import subprocess
    import tempfile

    cases = []
    for _, case, thinking, effort, _ in runs:
        ctx = {"thinking_mode": thinking, "enable_thinking": thinking == "thinking"}
        if effort is not None:
            ctx["reasoning_effort"] = effort
        cases.append({"messages": normalize(case["messages"]), "tools": case.get("tools"), "context": ctx})
    with tempfile.TemporaryDirectory() as d:
        json.dump(cases, open(os.path.join(d, "cases.json"), "w"), ensure_ascii=False)
        subprocess.run([os.path.join(HERE, "render_native"), os.path.join(HERE, "chat_template.jinja"),
                        os.path.join(d, "cases.json"), os.path.join(d, "out.json")], check=True)
        return json.load(open(os.path.join(d, "out.json")))


def main():
    template = open(os.path.join(HERE, "chat_template.jinja")).read()
    failed = 0
    runs = [(n, c, t, e, None) for n, c, t, e in CASES] + list(fixtures())
    engines = [("jinja2", None)]
    if "--native" in sys.argv:
        engines.append(("native", native(runs)))
    for engine, rendered in engines:
        for k, (name, case, thinking, effort, want) in enumerate(runs):
            name = f"{engine}: {name}"
            if want is None:
                want = expected(case, thinking, effort)
            if rendered is not None:
                got = rendered[k]
                if isinstance(got, dict):
                    print(f"FAIL {name}: {got['error']}")
                    failed += 1
                    continue
            else:
                try:
                    got = render(template, normalize(case["messages"]), case.get("tools"), thinking, effort)
                except jinja2.TemplateError as e:
                    print(f"FAIL {name}: {type(e).__name__}: {e}")
                    failed += 1
                    continue
            if got == want:
                print(f"ok   {name}")
                continue
            failed += 1
            at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            print(f"FAIL {name}: first difference at {at}\n  want: {want[max(0, at - 80):at + 80]!r}\n  got:  {got[max(0, at - 80):at + 80]!r}")
    total = len(runs) * len(engines)
    print(f"{total - failed}/{total} match")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

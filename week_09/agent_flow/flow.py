"""Plan, execute, verify and update memory until STOP or the turn limit."""
import ast
import json
import re
from . import prompts


def request(role, **context):
    if role in ("planner", "executor", "verifier", "generator"):
        labels = {"question": "Query", "tools": "Toolbox Metadata",
                  "sub_goal": "Sub-Goal", "tool": "Tool Name",
                  "metadata": "Toolbox Metadata", "context": "Relevant Data",
                  "memory": {"planner": "Previous Steps", "verifier": "Memory (Tools Used & Results)",
                             "generator": "Actions Taken"}.get(role, "Memory")}
        sections = []
        for name, value in context.items():
            if name == "tools":
                sections.append("Available Tools: " + ", ".join(value))
            text = json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else value
            sections.append(f"{labels.get(name, name)}:\n{text}")
        content = "\n\n".join(sections)
    else:
        content = json.dumps(context, ensure_ascii=False)
    return [{"role": "system", "content": prompts.PROMPTS[role]},
            {"role": "user", "content": content}]


def parse_plan(text):
    cleaned = text.replace("**", "")
    match = re.search(r"Context:\s*(.*?)\s*Sub-Goal:\s*(.*?)\s*Tool Name:\s*([^\n]+)",
                      cleaned, re.S | re.I)
    if not match:
        raise ValueError("Planner must return Context, Sub-Goal and Tool Name.")
    context, goal, tool = (value.strip() for value in match.groups())
    if tool not in prompts.TOOLS or not goal:
        raise ValueError("Planner must choose one available tool and a nonempty sub-goal.")
    return dict(context=context, sub_goal=goal, tool=tool)


def parse_command(text, tool):
    code = text.split("Generated Command:")[-1].strip()
    code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code).strip()
    try:
        tree = ast.parse(code)
        statement, = tree.body
        call = statement.value
        valid = (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                 and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "execution"
                 and isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                 and isinstance(call.func.value, ast.Name) and call.func.value.id == "tool"
                 and call.func.attr == "execute" and not call.args)
        if not valid:
            raise ValueError
        arguments = {arg.arg: ast.literal_eval(arg.value) for arg in call.keywords}
        allowed = {"query", "url"} if tool == "Web Search" else {"query"}
        if (len(arguments) != len(call.keywords) or not set(arguments) <= allowed
                or not isinstance(arguments.get("query"), str) or not arguments["query"].strip()):
            raise ValueError
        if tool == "Web Search" and not isinstance(arguments.get("url"), str):
            raise ValueError
    except (SyntaxError, ValueError, TypeError, AttributeError):
        raise ValueError("Expected one execution = tool.execute(...) call with literal tool arguments.") from None
    return arguments


def run_flow(questions, settings, generate, execute_tool, record_planner=None, execute_tools=None):
    """The inference inputs contain questions only, never reference answers."""
    states = [dict(question=q, trajectory=[]) for q in questions]
    active = list(range(len(states)))
    for turn in range(1, settings["max_turns"] + 1):
        plans = generate([request("planner", question=states[i]["question"],
                                  tools=prompts.TOOLS, memory=states[i]["trajectory"])
                          for i in active], settings["temperature"], settings["planner_max_new_tokens"])
        executable = []
        for i, generated in zip(active, plans):
            if record_planner is not None:
                record_planner(i, generated)
            entry = {"planner": generated["text"]}
            states[i]["trajectory"].append(entry)
            try:
                action = parse_plan(generated["text"])
                executable.append((i, action))
            except ValueError as error:
                entry["result"] = {"error": str(error)}
                entry["verifier"] = "Invalid action. Conclusion: CONTINUE"
        commands = generate([request("executor", question=states[i]["question"],
                                     **action, metadata=prompts.TOOLS[action["tool"]])
                             for i, action in executable], 0.0, settings["module_max_new_tokens"])
        pending_tools = []
        for (i, action), generated in zip(executable, commands):
            entry = states[i]["trajectory"][-1]
            entry["command"] = generated["text"]
            try:
                arguments = parse_command(generated["text"], action["tool"])
            except ValueError as error:
                entry["result"] = {"error": str(error)}
            else:
                if execute_tools is None:
                    entry["result"] = execute_tool(action["tool"], arguments)
                else:
                    pending_tools.append((i, action["tool"], arguments))
        if pending_tools:
            results = execute_tools([(tool, arguments) for _, tool, arguments in pending_tools])
            for (i, _, _), result in zip(pending_tools, results):
                states[i]["trajectory"][-1]["result"] = result
        reviews = generate([request("verifier", question=states[i]["question"],
                                    tools=prompts.TOOLS, memory=states[i]["trajectory"])
                            for i, _ in executable], 0.0, settings["module_max_new_tokens"])
        stopped = set()
        for (i, _), generated in zip(executable, reviews):
            text = generated["text"]
            states[i]["trajectory"][-1]["verifier"] = text
            if re.search(r"Conclusion:\s*STOP\s*$", text.replace("**", "")):
                states[i]["stop_reason"] = "verifier"
                stopped.add(i)
        active = [i for i in active if i not in stopped]
        if not active:
            break
    finals = generate([request("generator", question=s["question"], memory=s["trajectory"])
                       for s in states], 0.0, settings["max_new_tokens"])
    for state, answer in zip(states, finals):
        state.setdefault("stop_reason", "turn_limit")
        state.update(response=answer["text"], input_tokens=answer["input_tokens"],
                     output_tokens=answer["output_tokens"], finish_reason=answer["finish_reason"])
    return states

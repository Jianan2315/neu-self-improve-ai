"""Five AgentFlow tools; network services return evidence to the agent loop."""
import ast
import json
import os
import re
import time
from .flow import request


def api_json(url, payload, headers, service):
    import requests
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise RuntimeError(f"{service} connection failed.") from None
        else:
            if response.ok:
                return response.json()
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"{service} HTTP {response.status_code}; check API access or quota.")
        time.sleep(2 ** attempt)


def python_code(text):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = match.group(1).strip() if match else text.strip()
    tree = ast.parse(code)
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    allowed = {"print", "range", "sum", "min", "max", "abs", "round", "len", "int", "float",
               "list", "tuple", "sorted", "enumerate", "zip", "all", "any", "pow", "divmod"} | functions
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute, ast.ClassDef, ast.Global, ast.Nonlocal)):
            raise ValueError("Python Coder allows self-contained arithmetic only.")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Private Python names are unavailable.")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in allowed):
            raise ValueError("Unsupported Python function call.")
    return code


class Toolset:
    def __init__(self, generate, settings, run_python):
        self.generate = generate
        self.settings = settings
        self.run_python = run_python

    def ask(self, role, **context):
        return self.generate([request(role, **context)], 0.0,
                             self.settings["module_max_new_tokens"])[0]["text"]

    def execute(self, name, arguments):
        if name == "Base Generator":
            return self.ask("base_generator", **arguments)
        if name == "Python Coder":
            generated = self.ask("python_coder", **arguments)
            try:
                code = python_code(generated)
            except (ValueError, SyntaxError) as error:
                return {"code": generated, "error": str(error)}
            return {"code": code, **self.run_python(code)}
        if name == "OpenAI Search":
            return self.search(**arguments)
        if name == "Wikipedia Search":
            return self.wikipedia(**arguments)
        return self.web(**arguments)

    def search(self, query):
        body = api_json(
            "https://api.openai.com/v1/responses",
            {"model": self.settings["search_model"], "input": query,
             "instructions": "Search the web and return concise factual evidence relevant to the query, with source citations. State when evidence is insufficient.",
             "tools": [{"type": "web_search"}], "tool_choice": "required",
             "reasoning": {"effort": "low"}, "store": False},
            {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}, "OpenAI Search")
        if body.get("status") != "completed":
            raise RuntimeError("OpenAI Search did not complete.")
        output = body["output"]
        if not any(item["type"] == "web_search_call" and item.get("status") == "completed" for item in output):
            raise RuntimeError("OpenAI Search returned no completed web search.")
        parts = [part for item in output if item["type"] == "message"
                 for part in item.get("content", []) if part["type"] == "output_text"]
        text = "\n".join(part["text"] for part in parts)
        if not text.strip():
            raise RuntimeError("OpenAI Search returned no evidence text.")
        sources = {citation["url"]: {"url": citation["url"], "title": citation.get("title", "")}
                   for part in parts for citation in part.get("annotations", [])
                   if citation["type"] == "url_citation"}
        result = {"text": text, "sources": list(sources.values())}
        if not sources:
            result["warning"] = "No grounded web sources were returned; this is not verified search evidence."
        return result

    def wikipedia(self, query):
        import requests
        from email.utils import parsedate_to_datetime

        # One initial request, then at most three retries for HTTP 429.
        for attempt in range(4):
            response = requests.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "list": "search", "srsearch": query, "srlimit": 10, "format": "json"},
                headers={"User-Agent": "AgentFlowCourseProject/1.0 (Bamboogle evaluation)"}, timeout=60)
            if response.status_code != 429 or attempt == 3:
                break
            delay = 5 * 2 ** attempt
            retry_after = response.headers.get("Retry-After", "").strip()
            if retry_after.isdigit():
                delay = int(retry_after)
            elif retry_after:
                try:
                    delay = max(0, parsedate_to_datetime(retry_after).timestamp() - time.time())
                except (ValueError, TypeError, OverflowError):
                    pass
            response.close()
            time.sleep(delay)
        response.raise_for_status()
        pages = response.json()["query"]["search"]
        if not pages:
            return {"pages": [], "message": "No Wikipedia pages matched the query."}
        selection = self.ask("page_selector", query=query,
                             pages=[{"index": i, "title": p["title"]} for i, p in enumerate(pages)])
        try:
            indices = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", selection.strip()))
            if (not isinstance(indices, list) or len(indices) > 3
                    or any(type(i) is not int or not 0 <= i < len(pages) for i in indices)):
                raise ValueError
        except ValueError:
            return {"error": "Wikipedia page selector did not return a valid index list.", "selection": selection}
        from urllib.parse import quote
        return {"pages": [self.web(query, "https://en.wikipedia.org/wiki/" + quote(pages[i]["title"].replace(" ", "_")))
                          for i in dict.fromkeys(indices)]}

    def web(self, query, url):
        import ipaddress
        import socket
        import requests
        import numpy as np
        from bs4 import BeautifulSoup
        from urllib.parse import urlsplit, urljoin

        # Validate page URLs and redirects before fetching model-selected content.
        for _ in range(6):
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
                return {"url": url, "error": "Expected a public HTTP(S) page URL."}
            try:
                addresses = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
                if any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
                    return {"url": url, "error": "Page URL must use a public address."}
                response = requests.get(url, headers={"User-Agent": "AgentFlowCourseProject/1.0"},
                                        timeout=60, allow_redirects=False)
            except (OSError, requests.RequestException):
                return {"url": url, "error": "Could not retrieve this page."}
            if response.is_redirect:
                url = urljoin(url, response.headers["Location"])
                continue
            break
        else:
            return {"url": url, "error": "Too many page redirects."}
        if not response.ok:
            return {"url": url, "error": f"Page HTTP {response.status_code}."}
        if not any(t in response.headers.get("Content-Type", "") for t in ("text/html", "text/plain")):
            return {"url": url, "error": "This tool requires a text or HTML page."}
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        words = soup.get_text(" ", strip=True).split()[:self.settings["page_max_words"]]
        width, overlap = self.settings["chunk_words"], self.settings["chunk_overlap_words"]
        chunks = [" ".join(words[i:i + width]) for i in range(0, len(words), width - overlap)]
        if not chunks:
            return {"url": url, "error": "No readable page text."}
        texts, vectors = [query] + chunks, []
        for offset in range(0, len(texts), 64):
            body = api_json("https://api.openai.com/v1/embeddings",
                            {"model": self.settings["embedding_model"], "input": texts[offset:offset + 64]},
                            {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}, "OpenAI embeddings")
            vectors.extend(row["embedding"] for row in sorted(body["data"], key=lambda r: r["index"]))
        matrix = np.asarray(vectors, dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        scores = matrix[1:] @ matrix[0]
        indices = np.argsort(-scores, kind="stable")[:self.settings["retrieval_top_k"]]
        passages = [chunks[int(i)] for i in indices]
        return {"url": url, "text": self.ask("summarizer", query=query, url=url, passages=passages)}

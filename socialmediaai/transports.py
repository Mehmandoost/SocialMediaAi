from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from socialmediaai.errors import MissingCredentialError
from socialmediaai.providers.registry import ProviderProfile


def chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def anthropic_messages(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key, base_url=base_url or None)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [block.text for block in message.content if hasattr(block, "text")]
    return "".join(parts)


def codex_responses(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    return str(response)


def bedrock_converse(
    *,
    model: str,
    prompt: str,
    region: str | None = None,
) -> str:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("Install socialmediaai[bedrock] for AWS Bedrock support.") from exc

    client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))
    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 1024},
    }
    response = client.converse(modelId=model, messages=body["messages"], inferenceConfig=body["inferenceConfig"])
    output = response.get("output", {}).get("message", {}).get("content", [])
    texts = [part.get("text", "") for part in output if isinstance(part, dict)]
    return "".join(texts)


def vertex_chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    service_account_path: str | None = None,
) -> str:
    try:
        import google.auth.transport.requests
        from google.oauth2 import service_account
    except ImportError as exc:
        raise ImportError("Install socialmediaai[vertex] for Vertex AI support.") from exc

    if service_account_path:
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    else:
        import google.auth

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    credentials.refresh(google.auth.transport.requests.Request())
    token = credentials.token
    from openai import OpenAI

    url = base_url or "https://aiplatform.googleapis.com/v1"
    client = OpenAI(api_key=token, base_url=url)
    return chat_completions(base_url=url, api_key=token, model=model, prompt=prompt)


def copilot_http(
    *,
    base_url: str,
    token: str,
    model: str,
    prompt: str,
) -> str:
    import urllib.request

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode())
    return data["choices"][0]["message"]["content"]


def copilot_acp(
    *,
    prompt: str,
    command: str | None = None,
    args: tuple[str, ...] | None = None,
) -> str:
    binary = command or os.environ.get("HERMES_COPILOT_ACP_COMMAND") or os.environ.get("COPILOT_CLI_PATH") or "copilot"
    acp_args = args or tuple(
        (os.environ.get("HERMES_COPILOT_ACP_ARGS") or "--acp --stdio").split()
    )
    proc = subprocess.run(
        [binary, *acp_args],
        input=json.dumps({"prompt": prompt}).encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode() or f"copilot-acp exited {proc.returncode}")
    return proc.stdout.decode().strip()


def complete_with_profile(
    profile: ProviderProfile,
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    prompt: str,
    config_extras: dict[str, Any] | None = None,
) -> str:
    extras = config_extras or {}
    mode = profile.api_mode
    auth = profile.auth_type

    if auth == "external_process" or profile.name == "copilot-acp":
        return copilot_acp(prompt=prompt)

    if auth == "copilot":
        token = api_key
        if not token:
            raise MissingCredentialError("COPILOT_GITHUB_TOKEN", profile.name)
        return copilot_http(
            base_url=base_url or "https://api.githubcopilot.com",
            token=token,
            model=model,
            prompt=prompt,
        )

    if mode == "bedrock_converse" or auth == "aws_sdk":
        return bedrock_converse(model=model, prompt=prompt, region=extras.get("aws_region"))

    if auth == "vertex":
        return vertex_chat(
            base_url=base_url,
            model=model,
            prompt=prompt,
            service_account_path=extras.get("vertex_service_account"),
        )

    if (not api_key or not str(api_key).strip()) and auth not in (
        "oauth_external",
        "oauth_device_code",
        "aws_sdk",
        "external_process",
        "vertex",
        "copilot",
    ):
        env_name = profile.env_vars[0] if profile.env_vars else "API_KEY"
        raise MissingCredentialError(env_name, profile.name)

    if mode == "anthropic_messages":
        return anthropic_messages(base_url=base_url, api_key=api_key or "", model=model, prompt=prompt)

    if mode == "codex_responses":
        return codex_responses(base_url=base_url, api_key=api_key or "", model=model, prompt=prompt)

    return chat_completions(base_url=base_url, api_key=api_key or "", model=model, prompt=prompt)

# SocialMediaAi

Python library with project-wide configuration and LLM provider support (Hermes-compatible provider catalog).

```python
from socialmediaai import Config, Llm

config = Config()
llm = Llm(config)
llm.save_api_key("deepseek", "sk-...")
llm.set_model("deepseek", "deepseek-chat")
print(llm.complete("Hello"))
```

Secrets live in `~/.socialmediaai/.env` (override with `SMAI_HOME`). Non-secret settings live in `config.yaml` in the same directory.

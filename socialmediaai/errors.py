class MissingCredentialError(RuntimeError):
    """Raised when a provider call has no API key or token configured."""

    def __init__(self, env_var: str, provider: str | None = None) -> None:
        self.env_var = env_var
        self.provider = provider
        msg = f"Missing credential: set {env_var}"
        if provider:
            msg += f" for provider '{provider}'"
        super().__init__(msg)


class ConfigError(ValueError):
    """Invalid configuration write or read."""

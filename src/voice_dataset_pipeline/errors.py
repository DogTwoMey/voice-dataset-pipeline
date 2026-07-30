class VoiceDatasetError(RuntimeError):
    """Expected user-facing pipeline error."""


class ConfigurationError(VoiceDatasetError):
    """Invalid or incomplete project configuration."""


class ExternalToolError(VoiceDatasetError):
    """External executable or model process failed."""

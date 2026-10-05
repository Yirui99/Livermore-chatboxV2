"""Errors with a one-line, human-readable message and a hint. The CLI prints these without
a traceback; Streamlit and the server show the same text."""


class LivermoreError(Exception):
    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self):
        return self.message + (f"\n  hint: {self.hint}" if self.hint else "")


class ConfigError(LivermoreError):
    pass


class IndexNotFound(LivermoreError):
    pass


class NotesNotFound(LivermoreError):
    pass


class ModelFileError(LivermoreError):
    """A model file is missing, truncated, or fails its checksum."""


class ModelUnavailable(LivermoreError):
    """A model is not on disk and could not be downloaded."""


class BackendUnavailable(LivermoreError):
    pass

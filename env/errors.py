"""Custom exceptions for the customer support environment."""


class EnvError(Exception):
    """Base exception for all environment errors."""


class EnvironmentNotResetError(EnvError):
    """Raised when step() is called before reset()."""


class EnvironmentDoneError(EnvError):
    """Raised when step() is called after the episode has ended."""

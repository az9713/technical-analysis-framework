"""Custom exception hierarchy for the TA framework."""


class TAFrameworkError(Exception):
    """Base exception for all framework errors."""


class DataError(TAFrameworkError):
    """Errors related to data fetching, parsing, or validation."""


class InvalidSymbolError(DataError):
    """Raised when a symbol cannot be found or is invalid."""


class DataQualityError(DataError):
    """Raised when data fails quality checks."""


class InsufficientDataError(DataError):
    """Raised when not enough data is available for a computation."""


class IndicatorError(TAFrameworkError):
    """Errors related to indicator computation."""


class InvalidParameterError(IndicatorError):
    """Raised when an indicator receives invalid parameters."""


class SignalError(TAFrameworkError):
    """Errors related to signal generation."""


class BacktestError(TAFrameworkError):
    """Errors related to backtesting."""


class ConfigError(TAFrameworkError):
    """Errors related to configuration."""


class RegistryError(TAFrameworkError):
    """Errors related to the plugin registry."""

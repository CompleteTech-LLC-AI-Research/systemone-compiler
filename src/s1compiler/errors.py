class S1Error(Exception):
    """User-actionable framework error."""


class ConfigurationError(S1Error):
    pass


class DataError(S1Error):
    pass


class CandidateError(S1Error):
    pass


class BackendError(S1Error):
    """Provider failure. `transient` is set only by a backend that knows a retry may succeed."""

    transient = False


class BudgetExceeded(S1Error):
    pass

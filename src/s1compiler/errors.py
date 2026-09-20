class S1Error(Exception):
    """User-actionable framework error."""


class ConfigurationError(S1Error):
    pass


class DataError(S1Error):
    pass


class CandidateError(S1Error):
    pass


class BackendError(S1Error):
    pass


class BudgetExceeded(S1Error):
    pass

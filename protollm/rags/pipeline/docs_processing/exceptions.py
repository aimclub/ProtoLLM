class PathIsNotAssigned(Exception):
    def __init__(self, message):
        super.__init__(message)


class PipelineError(Exception):
    def __init__(self, message):
        super().__init__(message)


class FileExtensionError(Exception):
    def __init__(self, message):
        super().__init__(message)


class TransformerNameError(Exception):
    def __init__(self, message):
        super().__init__(message)


class LoaderNameError(Exception):
    def __init__(self, message):
        super().__init__(message)

"""mlx.core.config: configuration options"""



class ScopedUpdate:
    def __enter__(self) -> ScopedUpdate: ...

    def __exit__(self, *args) -> None: ...

def update(name: str, value: int) -> None:
    """
    Set the value of a process-wide configuration.

    See :ref:`environment_variables` for available configurations.

    Args:
        name (str): Name of configuration.
        value (int): Value of configuration.
    """

def get(name: str, default_value: int) -> int:
    """
    Read the value of a configuration. The lookup order is:

    1. The value set by :func:`mlx.core.config.scoped_update`.
    2. The value set by :func:`mlx.core.config.update`.
    3. The environment variable.
    4. The ``default_value``.

    See :ref:`environment_variables` for available configurations.

    Args:
        name (str): Name of configuration.
        default_value (int): Fallback value to return when not set.
    """

def scoped_update(**configs: int) -> ScopedUpdate:
    """
    Create a context manager to set configurations for current thread.

    See :ref:`environment_variables` for available configurations.

    Args:
        **configs (dict(str, int)): Configuration names and values as keyword arguments.
    """

"""mlx.cuda"""



def is_available() -> bool:
    """Check if the CUDA back-end is available."""

def get_arch_list() -> list[str]:
    """
    Get the CUDA architectures MLX was compiled for.

    Architectures use NVIDIA's ``sm_*`` (binary) and ``compute_*`` (PTX)
    notation, for example ``["sm_80", "sm_120a", "compute_120"]``.

    Returns:
        list(str): The compiled architectures, or an empty list if CUDA
        backend is not enabled.
    """

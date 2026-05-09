import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import sysconfig

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_SRC_DIR, "coalesced_collectives.cpp")
_MAKEFILE = os.path.join(_SRC_DIR, "Makefile")
_EXT_NAME = "_coalesced_collectives"
_META_VERSION = 1


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extension_suffix():
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        suffix = ".so"
    return suffix


def _torch_abi_flag(torch_module):
    abi = getattr(torch_module._C, "_GLIBCXX_USE_CXX11_ABI", None)
    if abi is None:
        return None
    return int(bool(abi))


def _quote_flags(flags):
    return " ".join(shlex.quote(str(flag)) for flag in flags)


def _build_config():
    import torch
    from torch.utils import cpp_extension

    py_include = sysconfig.get_paths().get("include")
    include_dirs = list(cpp_extension.include_paths())
    if py_include:
        include_dirs.append(py_include)

    lib_dirs = list(cpp_extension.library_paths())
    libs = ["c10", "torch", "torch_cpu", "torch_python"]

    cxx_flags = [
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        f"-DTORCH_EXTENSION_NAME={_EXT_NAME}",
    ]

    pybind11_flags = getattr(cpp_extension, "_get_pybind11_abi_build_flags", None)
    if pybind11_flags is not None:
        cxx_flags.extend(pybind11_flags())

    glibcxx_flags = getattr(cpp_extension, "_get_glibcxx_abi_build_flags", None)
    added_glibcxx_abi = False
    if glibcxx_flags is not None:
        extra_flags = glibcxx_flags()
        cxx_flags.extend(extra_flags)
        added_glibcxx_abi = any("_GLIBCXX_USE_CXX11_ABI" in flag for flag in extra_flags)

    abi = _torch_abi_flag(torch)
    if abi is not None and not added_glibcxx_abi:
        cxx_flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={abi}")
    cxx_flags.extend(f"-I{path}" for path in include_dirs)

    ld_flags = ["-shared"]
    ld_flags.extend(f"-L{path}" for path in lib_dirs)
    ld_flags.extend(f"-Wl,-rpath,{path}" for path in lib_dirs)
    ld_flags.extend(f"-l{name}" for name in libs)
    if sys.platform == "darwin":
        ld_flags.extend(["-undefined", "dynamic_lookup"])

    signature = {
        "meta_version": _META_VERSION,
        "source_sha256": _file_sha256(_SOURCE),
        "python_cache_tag": sys.implementation.cache_tag,
        "extension_suffix": _extension_suffix(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_cxx11_abi": abi,
        "cxx": os.environ.get("CXX", "c++"),
        "cxx_flags": cxx_flags,
        "ld_flags": ld_flags,
    }

    build_key = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    return cxx_flags, ld_flags, signature, build_key


def _paths(signature):
    so_path = os.path.join(_SRC_DIR, _EXT_NAME + signature["extension_suffix"])
    meta_path = os.path.join(_SRC_DIR, ".build_meta.json")
    lock_path = os.path.join(_SRC_DIR, ".compile.lock")
    return _SRC_DIR, so_path, meta_path, lock_path


def _metadata_matches(meta_path, signature):
    try:
        with open(meta_path) as fh:
            return json.load(fh) == signature
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _write_json_atomic(path, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(data, fh, sort_keys=True, indent=2)
        fh.write("\n")
    os.replace(tmp_path, path)


def _do_build(cxx_flags, ld_flags, tmp_so_path):
    """Run the actual native compilation via make."""
    env = os.environ.copy()
    env["COMM_CXXFLAGS"] = _quote_flags(cxx_flags)
    env["COMM_LDFLAGS"] = _quote_flags(ld_flags)
    env.setdefault("PYTHON", sys.executable)

    subprocess.run(
        ["make", "-f", _MAKEFILE, f"SRC={_SOURCE}", f"OUT={tmp_so_path}"],
        cwd=_SRC_DIR,
        env=env,
        check=True,
    )


def ensure_comm_extension():
    """Build the native helper if it is missing or stale, then return its path.

    When torch.distributed is initialized, uses a rank-0-first + barrier pattern: only rank 0 compiles, other ranks
    read the cached .so after the barrier. This avoids NFS/Lustre lock issues
    and guarantees exactly one compilation across the job.
    """
    cxx_flags, ld_flags, signature, build_key = _build_config()
    build_dir, so_path, meta_path, lock_path = _paths(signature)
    os.makedirs(build_dir, exist_ok=True)

    if os.path.exists(so_path) and _metadata_matches(meta_path, signature):
        return so_path

    import torch
    if not torch.distributed.is_initialized():
        # No distributed training, build locally with a file lock.
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

            if os.path.exists(so_path) and _metadata_matches(meta_path, signature):
                return so_path

            tmp_so_path = so_path + ".tmp"
            if os.path.exists(tmp_so_path):
                os.remove(tmp_so_path)

            _do_build(cxx_flags, ld_flags, tmp_so_path)

            os.replace(tmp_so_path, so_path)
            _write_json_atomic(meta_path, signature)
            return so_path

    # Distributed case: rank-0-first + barrier
    rank = torch.distributed.get_rank()

    if rank == 0:
        tmp_so_path = so_path + ".tmp"
        if os.path.exists(tmp_so_path):
            os.remove(tmp_so_path)

        try:
            _do_build(cxx_flags, ld_flags, tmp_so_path)
            os.replace(tmp_so_path, so_path)
            _write_json_atomic(meta_path, signature)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to build comm extension with make. "
                f"Build directory: {build_dir}"
            ) from exc

    torch.distributed.barrier()

    if rank != 0:
        if not os.path.exists(so_path):
            raise RuntimeError(
                f"Rank {rank} expected {so_path} to exist after rank-0 build. "
                f"Ensure all ranks share the same {build_dir}."
            )
        if not _metadata_matches(meta_path, signature):
            raise RuntimeError(
                f"Rank {rank} metadata mismatch for {so_path}. "
                f"Ensure all ranks share the same {build_dir}."
            )

    return so_path


def load_comm_extension():
    so_path = ensure_comm_extension()
    module_name = f"{__package__}.{_EXT_NAME}"
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {so_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module

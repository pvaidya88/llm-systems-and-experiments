import io
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional

from .ctx import SelectionContext


def worker_main(conn, context: Any, config: Dict[str, Any]) -> None:
    max_output_chars = int(config.get("max_output_chars", 4000))
    enable_note_yesno = bool(config.get("enable_note_yesno"))
    enable_rlm_query = bool(config.get("enable_rlm_query"))
    enable_ctx = bool(config.get("enable_ctx", True))
    cache_enabled = bool(config.get("cache_enabled", True))
    max_snippet_chars = int(config.get("max_snippet_chars", 200))
    memory_mb = config.get("memory_mb")
    cpu_seconds = config.get("cpu_seconds")

    _apply_resource_limits(memory_mb, cpu_seconds)

    def rpc_call(message: Dict[str, Any]) -> Any:
        conn.send(message)
        reply = conn.recv()
        if not isinstance(reply, dict):
            raise RuntimeError("Invalid RPC response.")
        if reply.get("error"):
            raise RuntimeError(str(reply["error"]))
        return reply.get("content")

    def llm_query(prompt: str, system: Optional[str] = None) -> str:
        return str(
            rpc_call({"type": "llm_query", "prompt": str(prompt), "system": system})
        )

    def llm_query_yesno(
        prompt: str, system: Optional[str] = None, max_retries: Optional[int] = None
    ) -> str:
        return str(
            rpc_call(
                {
                    "type": "llm_query_yesno",
                    "prompt": str(prompt),
                    "system": system,
                    "max_retries": max_retries,
                }
            )
        )

    def note_yesno(question: str) -> str:
        return str(rpc_call({"type": "note_yesno", "question": str(question)}))

    def rlm_query(
        question: str, sub_context: Any, depth_limit: int = 1
    ) -> str:
        return str(
            rpc_call(
                {
                    "type": "rlm_query",
                    "question": str(question),
                    "sub_context": sub_context,
                    "depth_limit": int(depth_limit),
                }
            )
        )

    def trace_event(event: Dict[str, Any]) -> None:
        rpc_call({"type": "ctx_event", "event": event})

    globals_dict: Dict[str, Any] = {
        "context": context,
        "llm_query": llm_query,
        "llm_query_yesno": llm_query_yesno,
    }
    if enable_note_yesno:
        globals_dict["note_yesno"] = note_yesno
    if enable_rlm_query:
        globals_dict["rlm_query"] = rlm_query
    if enable_ctx:
        globals_dict["ctx"] = SelectionContext(
            context,
            trace_hook=trace_event,
            max_snippet_chars=max_snippet_chars,
            cache_enabled=cache_enabled,
        )

    locals_dict: Dict[str, Any] = {}

    while True:
        msg = conn.recv()
        if not isinstance(msg, dict):
            continue
        msg_type = msg.get("type")
        if msg_type == "shutdown":
            break
        if msg_type == "get_var":
            name = msg.get("name")
            try:
                if name in locals_dict:
                    value = locals_dict[name]
                elif name in globals_dict:
                    value = globals_dict[name]
                else:
                    raise KeyError(f"Variable not found: {name}")
                conn.send({"type": "get_var_result", "content": str(value)})
            except Exception as exc:
                conn.send({"type": "get_var_result", "error": str(exc)})
            continue
        if msg_type != "exec":
            continue
        code = str(msg.get("code") or "")
        stdout = io.StringIO()
        stderr = io.StringIO()
        error_text = None
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, globals_dict, locals_dict)
        except Exception:
            error_text = traceback.format_exc()
        combined = (stdout.getvalue() + stderr.getvalue()).strip()
        if error_text:
            combined = (combined + "\n" + error_text).strip()
        if not combined:
            combined = "<no output>"
        truncated = False
        if len(combined) > max_output_chars:
            combined = combined[:max_output_chars] + "\n...<truncated>"
            truncated = True
        conn.send(
            {
                "type": "exec_result",
                "output": combined,
                "truncated": truncated,
                "error": error_text,
            }
        )


def _apply_resource_limits(memory_mb: Optional[int], cpu_seconds: Optional[int]) -> None:
    if os.name == "nt":
        _apply_windows_limits(memory_mb, cpu_seconds)
        return
    try:
        import resource
    except Exception:
        return
    if cpu_seconds is not None:
        limit = int(cpu_seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
    if memory_mb is not None:
        limit_bytes = int(memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def _apply_windows_limits(memory_mb: Optional[int], cpu_seconds: Optional[int]) -> None:
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.INT,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

    if cpu_seconds is not None:
        cpu_ticks = int(cpu_seconds * 10_000_000)
        try:
            info.BasicLimitInformation.PerProcessUserTimeLimit.QuadPart = cpu_ticks
        except Exception:
            info.BasicLimitInformation.PerProcessUserTimeLimit = cpu_ticks
        info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_TIME

    if memory_mb is not None:
        info.ProcessMemoryLimit = int(memory_mb) * 1024 * 1024
        info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY

    JobObjectExtendedLimitInformation = 9
    kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())

    # Keep job handle alive for the process lifetime.
    globals()["_JOB_OBJECT_HANDLE"] = job

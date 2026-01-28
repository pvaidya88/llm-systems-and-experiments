import unittest

from rlm_demo.repl import ReplEnvironment
from rlm_demo.tools import ToolRegistry


def noop_llm(prompt, system=None):
    return ""


class ToolRPCTest(unittest.TestCase):
    def test_tool_call(self):
        registry = ToolRegistry()
        registry.register("add", lambda a, b: a + b)
        repl = ReplEnvironment(
            context={},
            llm_query=noop_llm,
            enable_ctx=False,
            remote_tools_enabled=True,
            tool_registry=registry,
            repl_timeout_s=3.0,
        )
        try:
            result = repl.exec("print(tools.add(2, 3))")
            self.assertIn("5", result.get("output"))
        finally:
            repl.close()


if __name__ == "__main__":
    unittest.main()

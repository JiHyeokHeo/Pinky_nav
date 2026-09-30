"""Exercise watchdog method without opening serial ports or moving motors."""
import ast
from pathlib import Path
from types import SimpleNamespace


def test_retry_then_latch():
    tree = ast.parse((Path(__file__).parents[1]/'pinky_bringup/bringup.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Pinky')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_enforce_cmd_vel_timeout')
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<watchdog>', 'exec'), ns)
    class Stamp:
        def __init__(self, s): self.s = s
        def __sub__(self, other): return SimpleNamespace(nanoseconds=(self.s-other.s)*1e9)
    calls = []
    success = [False, True]
    def send(*args):
        calls.append(args)
        return success.pop(0)
    node = SimpleNamespace(last_cmd_vel_time=Stamp(0), cmd_vel_timed_out=False,
                           cmd_vel_timeout=.5, driver=SimpleNamespace(set_double_rpm=send),
                           get_logger=lambda: SimpleNamespace(error=lambda *a: None, warn=lambda *a: None))
    run = ns['_enforce_cmd_vel_timeout']
    run(node, Stamp(.4)); assert not calls
    run(node, Stamp(.6)); assert not node.cmd_vel_timed_out
    run(node, Stamp(.7)); assert node.cmd_vel_timed_out
    run(node, Stamp(.8)); assert calls == [(0, 0), (0, 0)]

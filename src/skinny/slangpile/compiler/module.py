from __future__ import annotations

import ast
import inspect
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from ..diagnostics import Diagnostic, SlangPileError
from ..registry import ExternFunction, get_shader, is_extern, is_shader
from ..types import SlangType, is_slang_type
from ..verification import VerificationOptions, VerificationResult, verify_file


@dataclass
class SourceMapEntry:
    generated_line: int
    source: str
    source_line: int
    symbol: str


@dataclass
class CompiledModule:
    module_name: str
    source: str
    imports: set[str] = field(default_factory=set)
    source_map: list[SourceMapEntry] = field(default_factory=list)

    @property
    def slang_module_name(self) -> str:
        return self.module_name.rsplit(".", 1)[-1]

    @property
    def relative_path(self) -> Path:
        return Path(*self.module_name.split(".")).with_suffix(".slang")

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        slang_path = out / self.relative_path
        slang_path.parent.mkdir(parents=True, exist_ok=True)
        slang_path.write_text(self.source, encoding="utf-8")
        map_path = slang_path.with_suffix(".slang.map.json")
        map_payload = {
            "generated": str(self.relative_path).replace("\\", "/"),
            "sources": sorted({entry.source for entry in self.source_map}),
            "mappings": [entry.__dict__ for entry in self.source_map],
        }
        map_path.write_text(json.dumps(map_payload, indent=2), encoding="utf-8")
        manifest = out / "slangpile_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "modules": [
                        {
                            "name": self.module_name,
                            "file": str(self.relative_path).replace("\\", "/"),
                            "imports": sorted(self.imports),
                        }
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return slang_path

    def verify(
        self,
        out_dir: str | Path,
        *,
        slangc: str | None = None,
        include_dirs: list[str | Path] | None = None,
        target: str | None = None,
        profile: str | None = None,
        entry_point: str | None = None,
    ) -> VerificationResult:
        slang_path = self.write(out_dir)
        options = VerificationOptions(
            slangc=slangc,
            include_dirs=[Path(p) for p in include_dirs or [out_dir]],
            target=target,
            profile=profile,
            entry_point=entry_point,
        )
        return verify_file(slang_path, options)

    def load_slangpy(
        self,
        out_dir: str | Path,
        *,
        device: object | None = None,
        include_paths: list[str | Path] | None = None,
        slangpy_module: object | None = None,
    ):
        from ..runtime import load_compiled_module

        return load_compiled_module(
            self,
            out_dir,
            device=device,
            include_paths=include_paths,
            slangpy_module=slangpy_module,
        )


class ModuleCompiler:
    def __init__(self, module: ModuleType):
        self.module = module
        self.module_name = module.__name__
        self.shaders = {
            name: value
            for name, value in vars(module).items()
            if is_shader(value) and get_shader(value).module_name == self.module_name
        }
        self.externs = {name: value for name, value in vars(module).items() if is_extern(value)}
        self.imports: set[str] = set()
        self.source_map: list[SourceMapEntry] = []

    def compile(self) -> CompiledModule:
        slang_module_name = self.module_name.rsplit(".", 1)[-1]
        lines = []
        function_sources: list[str] = []
        for name in sorted(self.shaders):
            function_sources.append(self._compile_function(name, self.shaders[name]))
        import_lines = [f"import {module};" for module in sorted(self.imports)]
        if import_lines:
            lines.extend(import_lines)
            lines.append("")
        lines.extend(function_sources)
        source = "\n\n".join(part.rstrip() for part in lines if part != "")
        source += "\n"
        return CompiledModule(self.module_name, source, set(self.imports), self.source_map)

    def _compile_function(self, name: str, fn: Any) -> str:
        shader = get_shader(fn)
        if shader is None:
            raise AssertionError("expected shader function")
        source_fn = shader.fn
        try:
            source = textwrap.dedent(inspect.getsource(source_fn))
        except OSError as exc:
            raise SlangPileError(Diagnostic("error", f"cannot read source for shader '{name}': {exc}")) from exc
        tree = ast.parse(source)
        func = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        if func is None:
            raise SlangPileError(Diagnostic("error", f"shader '{name}' is not a function"))
        return_type = self._resolve_annotation(func.returns, source_fn.__globals__)
        args = []
        for arg in func.args.args:
            args.append((arg.arg, self._resolve_annotation(arg.annotation, source_fn.__globals__)))
        body_lines = []
        local_types = {arg_name: arg_type for arg_name, arg_type in args}
        emitter = FunctionEmitter(self, source_fn.__globals__, local_types)
        for stmt in func.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            if isinstance(stmt, ast.Pass):
                continue
            body_lines.extend(emitter.emit_stmt(stmt))
        arg_source = ", ".join(f"{typ.slang_name} {arg_name}" for arg_name, typ in args)
        header = f"public {return_type.slang_name} {name}({arg_source})"
        source_file = inspect.getsourcefile(source_fn) or "<unknown>"
        self.source_map.append(SourceMapEntry(1, source_file, func.lineno, name))
        body = "\n".join(f"    {line}" for line in body_lines)
        return f"{header}\n{{\n{body}\n}}"

    def _resolve_annotation(self, node: ast.AST | None, globals_: dict[str, Any]) -> SlangType:
        if node is None:
            raise SlangPileError(Diagnostic("error", "shader arguments and returns require SlangPile type annotations"))
        value = _eval_annotation(node, globals_)
        if not is_slang_type(value):
            raise SlangPileError(Diagnostic("error", f"unsupported shader type annotation: {ast.unparse(node)}"))
        return value

    def resolve_call_name(self, node: ast.AST, globals_: dict[str, Any]) -> str:
        if isinstance(node, ast.Name):
            if node.id in self.shaders:
                return node.id
            value = globals_.get(node.id)
            if is_shader(value):
                shader = get_shader(value)
                if shader.module_name != self.module_name:
                    self.imports.add(shader.module_name)
                    return shader.name
                return shader.name
            if isinstance(value, ExternFunction):
                if value.module:
                    self.imports.add(value.module)
                return value.slang_name
            if node.id in {"min", "max", "abs", "sin", "cos", "tan", "sqrt", "dot", "normalize", "lerp"}:
                return node.id
        raise SlangPileError(Diagnostic("error", f"unsupported function call: {ast.unparse(node)}"))


class FunctionEmitter:
    def __init__(self, compiler: ModuleCompiler, globals_: dict[str, Any], local_types: dict[str, SlangType]):
        self.compiler = compiler
        self.globals = globals_
        self.local_types = local_types

    def emit_stmt(self, stmt: ast.stmt) -> list[str]:
        if isinstance(stmt, ast.Return):
            return [f"return {self.emit_expr(stmt.value)};"]
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                self.unsupported(stmt, "only simple local assignments are supported")
            return [f"auto {stmt.targets[0].id} = {self.emit_expr(stmt.value)};"]
        if isinstance(stmt, ast.AnnAssign):
            if not isinstance(stmt.target, ast.Name) or stmt.value is None:
                self.unsupported(stmt, "only initialized annotated local assignments are supported")
            typ = self.compiler._resolve_annotation(stmt.annotation, self.globals)
            self.local_types[stmt.target.id] = typ
            return [f"{typ.slang_name} {stmt.target.id} = {self.emit_expr(stmt.value)};"]
        if isinstance(stmt, ast.If):
            lines = [f"if ({self.emit_expr(stmt.test)})", "{"]
            for child in stmt.body:
                lines.extend(f"    {line}" for line in self.emit_stmt(child))
            lines.append("}")
            if stmt.orelse:
                lines.append("else")
                lines.append("{")
                for child in stmt.orelse:
                    lines.extend(f"    {line}" for line in self.emit_stmt(child))
                lines.append("}")
            return lines
        self.unsupported(stmt, f"unsupported statement: {type(stmt).__name__}")

    def emit_expr(self, expr: ast.AST | None) -> str:
        if expr is None:
            self.unsupported(ast.Pass(), "missing expression")
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return "true" if expr.value else "false"
            if isinstance(expr.value, (int, float)):
                return repr(expr.value)
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.BinOp):
            return f"({self.emit_expr(expr.left)} {self.binop(expr.op)} {self.emit_expr(expr.right)})"
        if isinstance(expr, ast.UnaryOp):
            return f"({self.unaryop(expr.op)}{self.emit_expr(expr.operand)})"
        if isinstance(expr, ast.Compare):
            if len(expr.ops) != 1 or len(expr.comparators) != 1:
                self.unsupported(expr, "chained comparisons are not supported")
            return f"({self.emit_expr(expr.left)} {self.cmpop(expr.ops[0])} {self.emit_expr(expr.comparators[0])})"
        if isinstance(expr, ast.IfExp):
            return f"({self.emit_expr(expr.test)} ? {self.emit_expr(expr.body)} : {self.emit_expr(expr.orelse)})"
        if isinstance(expr, ast.Call):
            name = self.compiler.resolve_call_name(expr.func, self.globals)
            args = ", ".join(self.emit_expr(arg) for arg in expr.args)
            return f"{name}({args})"
        if isinstance(expr, ast.Attribute):
            return f"{self.emit_expr(expr.value)}.{expr.attr}"
        self.unsupported(expr, f"unsupported expression: {type(expr).__name__}")

    def binop(self, op: ast.operator) -> str:
        table = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
        }
        for typ, symbol in table.items():
            if isinstance(op, typ):
                return symbol
        self.unsupported(op, f"unsupported binary operator: {type(op).__name__}")

    def unaryop(self, op: ast.unaryop) -> str:
        if isinstance(op, ast.USub):
            return "-"
        if isinstance(op, ast.UAdd):
            return "+"
        if isinstance(op, ast.Not):
            return "!"
        self.unsupported(op, f"unsupported unary operator: {type(op).__name__}")

    def cmpop(self, op: ast.cmpop) -> str:
        table = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        for typ, symbol in table.items():
            if isinstance(op, typ):
                return symbol
        self.unsupported(op, f"unsupported comparison operator: {type(op).__name__}")

    def unsupported(self, node: ast.AST, message: str):
        raise SlangPileError(Diagnostic("error", message, source="python", line=getattr(node, "lineno", None)))


def _eval_annotation(node: ast.AST, globals_: dict[str, Any]) -> Any:
    expr = ast.Expression(node)
    ast.fix_missing_locations(expr)
    return eval(compile(expr, "<slangpile-annotation>", "eval"), globals_)

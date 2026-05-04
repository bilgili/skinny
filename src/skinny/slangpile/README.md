# SlangPile

SlangPile is an embedded Python-to-Slang transpiler. Write GPU shader logic as decorated Python functions and structs, and SlangPile compiles them to valid [Slang](https://shader-slang.com/) source files (`.slang`). The generated code can then be compiled by `slangc` to SPIR-V for Vulkan compute pipelines.

## Why?

- **Python-side composability** -- generate shader variants, parameter sweeps, and material graphs from Python.
- **Unified codegen pipeline** -- the same toolchain handles both hand-authored and generated shaders.
- **Testability** -- unit-test transpilation output without a GPU.

## Quick Start

```python
from skinny import slangpile as sp

@sp.shader
def add(a: sp.float32, b: sp.float32) -> sp.float32:
    return a + b
```

Compile to Slang source:

```python
compiled = sp.compile_module("my_module")
print(compiled.source)
# Output:
# public float add(float a, float b)
# {
#     return (a + b);
# }
```

Write to disk and optionally verify with `slangc`:

```python
compiled.write("generated/")                           # writes generated/my_module.slang
result = compiled.verify("generated/")                 # runs slangc
assert result.ok, result.stderr
```

## CLI

SlangPile includes a command-line interface:

```bash
# Transpile a Python module to .slang
python -m skinny.slangpile build my_module -o generated/

# Transpile and verify with slangc
python -m skinny.slangpile build my_module -o generated/ --verify

# Check that a module transpiles without errors (no output)
python -m skinny.slangpile check my_module

# Verify an existing .slang file with slangc
python -m skinny.slangpile verify generated/my_module.slang -I shaders/
```

---

## Type System

SlangPile types map directly to Slang types:

| Python                | Slang         |
|-----------------------|---------------|
| `sp.float32`          | `float`       |
| `sp.float64`          | `double`      |
| `sp.float16`          | `half`        |
| `sp.int32`            | `int`         |
| `sp.uint32`           | `uint`        |
| `sp.bool`             | `bool`        |
| `sp.float32x2`        | `float2`      |
| `sp.float32x3`        | `float3`      |
| `sp.float32x4`        | `float4`      |
| `sp.int32x3`          | `int3`        |
| `sp.uint32x4`         | `uint4`       |
| `sp.float32x3x3`      | `float3x3`    |
| `sp.float32x4x4`      | `float4x4`    |

All scalar, vector (2/3/4), and matrix types for float16/32/64, int8/16/32/64, and uint8/16/32/64 are available.

---

## Functions

Decorate with `@sp.shader`. All parameters and the return type must have SlangPile type annotations.

```python
@sp.shader
def lerp_color(a: sp.float32x3, b: sp.float32x3, t: sp.float32) -> sp.float32x3:
    return a * (1.0 - t) + b * t
```

Generated Slang:

```slang
public float3 lerp_color(float3 a, float3 b, float t)
{
    return ((a * (1.0 - t)) + (b * t));
}
```

### Void return

Return `None` in the annotation for `void` functions:

```python
@sp.shader
def do_work(x: sp.float32) -> None:
    some_side_effect(x)
```

---

## Variables and Locals

### Annotated locals (explicit type)

```python
@sp.shader
def example(x: sp.float32) -> sp.float32:
    y: sp.float32 = x * 2.0          # float y = (x * 2.0);
    return y
```

### Inferred locals (var)

```python
@sp.shader
def example(x: sp.float32) -> sp.float32:
    y = x * 2.0                      # var y = (x * 2.0);
    return y
```

### Uninitialized locals

```python
@sp.shader
def example() -> sp.float32:
    result: sp.float32               # float result;
    result = 42.0
    return result
```

### Reassignment

Reassigning a variable does not re-declare it:

```python
@sp.shader
def example(x: sp.float32) -> sp.float32:
    y: sp.float32 = x
    y = y * 2.0                      # y = (y * 2.0);  (no redeclaration)
    return y
```

---

## Control Flow

### If / elif / else

```python
@sp.shader
def classify(x: sp.float32) -> sp.int32:
    if x > 1.0:
        return 1
    elif x < -1.0:
        return -1
    else:
        return 0
```

Generated:

```slang
if ((x > 1.0))
{
    return 1;
}
else if ((x < -1.0))
{
    return -1;
}
else
{
    return 0;
}
```

### For loops (range-based)

SlangPile supports `range()` with 1, 2, or 3 arguments:

```python
@sp.shader
def sum_n(n: sp.int32) -> sp.int32:
    total: sp.int32 = 0
    for i in range(n):                # for (int i = 0; i < n; i++)
        total += i
    return total
```

```python
for i in range(5):                    # for (int i = 0; i < 5; i++)
for i in range(2, 10):                # for (int i = 2; i < 10; i++)
for i in range(0, 100, 2):            # for (int i = 0; i < 100; i += 2)
```

Nested loops work naturally:

```python
for z in range(-1, 2):
    for y in range(-1, 2):
        for x in range(-1, 2):
            # 3x3x3 neighborhood
            pass
```

### While loops

```python
@sp.shader
def converge(x: sp.float32) -> sp.float32:
    while x > 0.001:
        x = x * 0.5
    return x
```

### Break and continue

```python
for i in range(100):
    if i > 50:
        break
    if i % 2 == 0:
        continue
    process(i)
```

---

## Operators

### Arithmetic

`+`, `-`, `*`, `/`, `%` -- all standard arithmetic operators.

### Comparison

`==`, `!=`, `<`, `<=`, `>`, `>=`

### Boolean

Python `and`/`or`/`not` map to Slang `&&`/`||`/`!`:

```python
if x > 0.0 and y < 1.0:              # if (((x > 0.0) && (y < 1.0)))
if not valid:                         # if ((!valid))
```

### Bitwise

`&`, `|`, `^`, `~`, `<<`, `>>`:

```python
flags: sp.uint32 = a & 0xFF
mask = flags | (1 << bit)
inverted = ~flags
```

### Augmented assignment

`+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`:

```python
total += delta
mask &= 0xFF
```

### Ternary (conditional expression)

Python's `x if cond else y` maps to Slang's `cond ? x : y`:

```python
sign: sp.float32 = 1.0 if x > 0.0 else -1.0
# float sign = ((x > 0.0) ? 1.0 : -1.0);
```

---

## Vector Constructors

Call SlangPile types as constructors:

```python
v = sp.float32x3(1.0, 2.0, 3.0)      # float3(1.0, 2.0, 3.0)
zero = sp.float32x3(0.0)              # float3(0.0)  -- broadcast
base = sp.int32x3(floor(position))    # int3(floor(position))  -- type cast
```

### Attribute access (swizzles)

Vector component access works via standard attribute syntax:

```python
x_val = position.x
uv = color.xy
rgb = color.rgb
```

### Array subscript

```python
value = buffer[i]                      # buffer[i]
buffer[idx] = new_value                # buffer[idx] = new_value;
```

---

## Built-in Functions

SlangPile recognizes 50+ Slang built-in functions and emits them as-is:

**Math:** `min`, `max`, `abs`, `sign`, `clamp`, `lerp`, `mix`, `step`, `smoothstep`, `saturate`, `frac`, `fmod`, `floor`, `ceil`, `round`, `mad`, `rcp`

**Trigonometry:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`

**Exponential:** `sqrt`, `pow`, `exp`, `exp2`, `log`, `log2`

**Vector:** `dot`, `cross`, `length`, `distance`, `normalize`, `reflect`, `refract`

**Logic:** `any`, `all`, `isnan`, `isinf`

**Derivatives:** `ddx`, `ddy`

**Type constructors:** `float`, `float2`, `float3`, `float4`, `int`, `int2`, `int3`, `int4`, `uint`, `uint2`, `uint3`, `uint4`, `half`, `half2`, `half3`, `half4`, `bool`

Example:

```python
@sp.shader
def shade(N: sp.float32x3, L: sp.float32x3) -> sp.float32:
    NdotL = max(dot(N, L), 0.0)
    return clamp(NdotL, 0.0, 1.0)
```

---

## Structs

Define Slang structs with `@sp.struct`. Fields use SlangPile type annotations.

### Simple data struct

```python
@sp.struct
class LightSample:
    point: sp.float32x3
    normal: sp.float32x3
    radiance: sp.float32x3
    pdfArea: sp.float32
    valid: sp.bool
```

Generated:

```slang
public struct LightSample
{
    float3 point;
    float3 normal;
    float3 radiance;
    float pdfArea;
    bool valid;
}
```

### Struct with interface conformance

```python
@sp.struct(conforms_to="ILight")
class DirectionalLight:
    dir: sp.float32x3
    rad: sp.float32x3
```

Generated:

```slang
public struct DirectionalLight : ILight
{
    float3 dir;
    float3 rad;
}
```

### Struct with methods

Methods are regular Python functions inside the class. The `self` parameter is automatically stripped -- field references use bare names in the generated Slang:

```python
@sp.struct(conforms_to="ISampler")
class LambertSampler:
    N: sp.float32x3

    def sampleDirection(self, u: sp.float32x2) -> sp.float32x3:
        return sampleCosineHemisphere(u, self.N)

    def pdf(self, L: sp.float32x3) -> sp.float32:
        return cosineHemispherePdf(L, self.N)
```

Generated:

```slang
public struct LambertSampler : ISampler
{
    float3 N;

    float3 sampleDirection(float2 u)
    {
        return sampleCosineHemisphere(u, N);
    }

    float pdf(float3 L)
    {
        return cosineHemispherePdf(L, N);
    }
}
```

### Mutating methods

Mark methods that modify struct fields with `@sp.mutating`:

```python
@sp.struct
class Counter:
    value: sp.int32

    @sp.mutating
    def increment(self) -> None:
        self.value += 1
```

Generated:

```slang
[mutating] void increment()
{
    value += 1;
}
```

### Nested struct fields

Structs can reference other struct types as field types:

```python
@sp.struct
class SkinLayerStack:
    epi: SkinLayer
    der: SkinLayer
    sub: SkinLayer
```

Access nested fields naturally:

```python
stack.epi.sigmaA = melaninAbsorption(melanin)
```

---

## Extern Declarations

Reference functions and types defined in other `.slang` modules.

### Extern functions

Declare a function signature without providing a body:

```python
pcgHash = sp.extern(
    name="pcgHash",
    module="common",
    args=[sp.uint32],
    returns=sp.uint32,
)
```

This tells SlangPile that `pcgHash` exists in `common.slang`. When used in a shader function, the compiler automatically adds `import common;` to the output.

Use extern functions like regular calls:

```python
@sp.shader
def hash_cell(cell: sp.uint32) -> sp.uint32:
    return pcgHash(cell)
```

### Extern constants

Module-level constants (like `PI`) can be declared as zero-argument externs:

```python
PI = sp.extern(name="PI", module="common", args=[], returns=sp.float32)
```

Then reference as a bare name in expressions:

```python
area = 4.0 * PI * r * r
```

### Extern types

Reference struct types defined in other modules:

```python
LightSample = sp.extern_type("LightSample")
Ray = sp.extern_type("Ray")
```

These can be used as parameter types, return types, and local variable types:

```python
@sp.shader
def trace(ray: Ray) -> LightSample:
    s: LightSample
    s.point = ray.origin
    return s
```

---

## Module Imports

Import hand-written `.slang` modules:

```python
_common = sp.slang_import("common")
_interfaces = sp.slang_import("interfaces")
```

This emits `import common;` and `import interfaces;` at the top of the generated file. The return value is a marker object -- assign it to a throwaway variable.

---

## Constants

Declare `static const` values:

```python
INK_ABS_MAX = sp.const("INK_ABS_MAX", sp.float32, 22.0)
MAX_STEPS = sp.const("MAX_STEPS", sp.int32, 128)
```

Generated:

```slang
static const float INK_ABS_MAX = 22.0;
static const int MAX_STEPS = 128;
```

Reference them as bare names in shader code:

```python
@sp.shader
def apply_ink(density: sp.float32) -> sp.float32:
    return density * INK_ABS_MAX
```

---

## Verbatim Slang

Inject raw Slang code for constructs SlangPile does not (yet) express:

```python
sp.verbatim("[[vk::binding(4)]] Sampler2D<float4> envMap;")
sp.verbatim("[[vk::binding(17)]] StructuredBuffer<SphereLight> sphereLights;")
```

The verbatim block is emitted after imports but before structs and functions. Use this for Vulkan descriptor bindings, `#define` macros, or any other raw Slang.

---

## Generic Functions

Define functions with generic type parameters constrained by interfaces:

```python
TA = sp.generic("TA", "ISampler")
TB = sp.generic("TB", "ISampler")

@sp.shader(generics={"TA": "ISampler", "TB": "ISampler"})
def misPrimaryWeight(primary: TA, companion: TB, L: sp.float32x3) -> sp.float32:
    return powerHeuristic(primary.pdf(L), companion.pdf(L))
```

Generated:

```slang
public float misPrimaryWeight<TA : ISampler, TB : ISampler>(TA primary, TB companion, float3 L)
{
    return powerHeuristic(primary.pdf(L), companion.pdf(L));
}
```

The `sp.generic()` call creates a type placeholder for use in annotations. The `generics` dict on `@sp.shader()` controls the `<T : Interface>` clause in the output.

---

## Parameter Modifiers: inout / out

Wrap parameter types to emit `inout` or `out` modifiers:

```python
@sp.shader
def swap(a: sp.inout(sp.float32), b: sp.inout(sp.float32)) -> None:
    t = a
    a = b
    b = t
```

Generated:

```slang
public void swap(inout float a, inout float b)
```

`sp.out()` works the same way for output-only parameters:

```python
@sp.shader
def buildBasis(N: sp.float32x3, T: sp.out(sp.float32x3), B: sp.out(sp.float32x3)) -> None:
    ...
```

---

## Compiled Module API

`sp.compile_module()` returns a `CompiledModule` with these members:

| Member             | Description                                             |
|--------------------|---------------------------------------------------------|
| `.source`          | The generated Slang source code as a string             |
| `.module_name`     | Python module name (e.g. `"samplers.ggx"`)              |
| `.slang_module_name` | Last component (e.g. `"ggx"`)                         |
| `.relative_path`   | Output path (e.g. `Path("samplers/ggx.slang")`)         |
| `.imports`         | Set of Slang module names that are imported             |
| `.source_map`      | List of source map entries for debugging                |
| `.write(out_dir)`  | Write `.slang`, `.map.json`, and manifest to disk       |
| `.verify(out_dir)` | Write and run `slangc` to check for compilation errors  |

### Verification

```python
result = compiled.verify("generated/", include_dirs=["shaders/"])
if not result.ok:
    for diag in result.diagnostics:
        print(diag.format())
```

`VerificationResult` fields: `.ok`, `.command`, `.stdout`, `.stderr`, `.diagnostics`.

### Source maps

Each `.write()` produces a `.slang.map.json` alongside the generated file, mapping generated line numbers back to the Python source:

```json
{
  "generated": "samplers/lambert.slang",
  "sources": ["shader_kernels/samplers/lambert.py"],
  "mappings": [
    {"generated_line": 1, "source": "...", "source_line": 10, "symbol": "sampleDirection"}
  ]
}
```

---

## Output Structure

The generated `.slang` file follows this order:

1. `import` statements (from `sp.slang_import` and extern module references)
2. Verbatim blocks (from `sp.verbatim`)
3. Static constants (from `sp.const`)
4. Struct definitions (alphabetical)
5. Function definitions (alphabetical)

All structs are emitted as `public struct` and all functions as `public`. Within a single module, forward references between functions work because Slang resolves all module-level declarations before checking bodies.

---

## Complete Example

A GGX microfacet sampler that conforms to the `ISampler` interface:

```python
from skinny import slangpile as sp

_common = sp.slang_import("common")
_interfaces = sp.slang_import("interfaces")

PI = sp.extern(name="PI", module="common", args=[], returns=sp.float32)
buildBasis = sp.extern(
    name="buildBasis", module="common",
    args=[sp.float32x3, sp.out(sp.float32x3), sp.out(sp.float32x3)],
    returns=sp.float32,
)


@sp.struct(conforms_to="ISampler")
class GGXSampler:
    N: sp.float32x3
    V: sp.float32x3
    roughness: sp.float32

    def sampleDirection(self, u: sp.float32x2) -> sp.float32x3:
        a: sp.float32 = max(self.roughness * self.roughness, 1e-4)
        a2: sp.float32 = a * a
        cosTheta: sp.float32 = sqrt((1.0 - u.y) / (1.0 + (a2 - 1.0) * u.y))
        sinTheta: sp.float32 = sqrt(max(0.0, 1.0 - cosTheta * cosTheta))
        phi: sp.float32 = 2.0 * PI * u.x

        T: sp.float32x3
        B: sp.float32x3
        buildBasis(self.N, T, B)

        H: sp.float32x3 = normalize(
            T * cos(phi) * sinTheta +
            B * sin(phi) * sinTheta +
            self.N * cosTheta
        )
        return reflect(-self.V, H)

    def pdf(self, L: sp.float32x3) -> sp.float32:
        H: sp.float32x3 = normalize(self.V + L)
        NdotH: sp.float32 = max(dot(self.N, H), 0.0)
        VdotH: sp.float32 = max(dot(self.V, H), 1e-6)
        a: sp.float32 = max(self.roughness * self.roughness, 1e-4)
        a2: sp.float32 = a * a
        d: sp.float32 = NdotH * NdotH * (a2 - 1.0) + 1.0
        D: sp.float32 = a2 / (PI * d * d)
        return D * NdotH / (4.0 * VdotH)
```

---

## Architecture

```
slangpile/
  __init__.py          # Public API exports
  api.py               # Decorators and module-level helpers
  types.py             # SlangType, ScalarType, VectorType, MatrixType definitions
  registry.py          # ShaderFunction, ExternFunction, StructDefinition storage
  compiler/
    __init__.py
    module.py           # ModuleCompiler + FunctionEmitter -- the transpiler core
  verification.py      # slangc invocation and diagnostic parsing
  diagnostics.py       # Diagnostic and SlangPileError types
  runtime.py           # SlangPy integration (optional GPU runtime)
  cli.py               # Command-line interface
```

### How transpilation works

1. `ModuleCompiler.__init__` scans the Python module's globals for `@sp.shader` functions, `@sp.struct` classes, `SlangImport`, `Verbatim`, `SlangConst`, and `ExternFunction` objects.
2. For each shader function and struct method, it reads the Python source via `inspect.getsource()`, parses it with `ast.parse()`, and walks the AST.
3. `FunctionEmitter` converts each AST node to Slang text: `ast.Assign` becomes variable declarations, `ast.If` becomes if/else blocks, `ast.For` (with `range()`) becomes C-style for loops, etc.
4. The emitter resolves names against the module's globals to distinguish between same-module calls, cross-module shader calls, extern functions, and built-in Slang functions.
5. The output is assembled in order: imports, verbatim, constants, structs, functions.

---

## Limitations

- **Range-based for only** -- `for` loops must use `range()`. Arbitrary iterables are not supported.
- **No `match`/`case`** -- Python 3.10 structural pattern matching is not transpiled to `switch`.
- **No closures or lambdas** -- all functions must be module-level `@sp.shader` or struct methods.
- **No default parameter values** -- all arguments are positional, no defaults.
- **No list/dict/set** -- only scalar, vector, matrix, and struct types.
- **No string operations** -- string constants emit as Slang string literals but are rarely useful.
- **No inheritance** -- structs support interface conformance (`conforms_to`) but not struct inheritance.
- **No operator overloading** -- Python dunder methods are not transpiled.
- **Integer literals** -- Python integer literals emit without `u` suffix; Slang handles implicit int-to-uint conversion. Explicit `uint(x)` casts work via the built-in function table.

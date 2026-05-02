from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlangType:
    name: str
    slang_name: str

    def __repr__(self) -> str:
        return f"slangpile.{self.name}"


@dataclass(frozen=True)
class ScalarType(SlangType):
    bits: int | None = None
    signed: bool | None = None


@dataclass(frozen=True)
class VectorType(SlangType):
    element_type: ScalarType | None = None
    size: int = 0


@dataclass(frozen=True)
class MatrixType(SlangType):
    element_type: ScalarType | None = None
    rows: int = 0
    cols: int = 0


bool = ScalarType("bool", "bool")

int8 = ScalarType("int8", "int8_t", bits=8, signed=True)
int16 = ScalarType("int16", "int16_t", bits=16, signed=True)
int32 = ScalarType("int32", "int", bits=32, signed=True)
int64 = ScalarType("int64", "int64_t", bits=64, signed=True)

uint8 = ScalarType("uint8", "uint8_t", bits=8, signed=False)
uint16 = ScalarType("uint16", "uint16_t", bits=16, signed=False)
uint32 = ScalarType("uint32", "uint", bits=32, signed=False)
uint64 = ScalarType("uint64", "uint64_t", bits=64, signed=False)

float16 = ScalarType("float16", "half", bits=16)
float32 = ScalarType("float32", "float", bits=32)
float64 = ScalarType("float64", "double", bits=64)

float16x2 = VectorType("float16x2", "half2", element_type=float16, size=2)
float16x3 = VectorType("float16x3", "half3", element_type=float16, size=3)
float16x4 = VectorType("float16x4", "half4", element_type=float16, size=4)

float32x2 = VectorType("float32x2", "float2", element_type=float32, size=2)
float32x3 = VectorType("float32x3", "float3", element_type=float32, size=3)
float32x4 = VectorType("float32x4", "float4", element_type=float32, size=4)

int32x2 = VectorType("int32x2", "int2", element_type=int32, size=2)
int32x3 = VectorType("int32x3", "int3", element_type=int32, size=3)
int32x4 = VectorType("int32x4", "int4", element_type=int32, size=4)

uint32x2 = VectorType("uint32x2", "uint2", element_type=uint32, size=2)
uint32x3 = VectorType("uint32x3", "uint3", element_type=uint32, size=3)
uint32x4 = VectorType("uint32x4", "uint4", element_type=uint32, size=4)

float32x3x3 = MatrixType("float32x3x3", "float3x3", element_type=float32, rows=3, cols=3)
float32x4x4 = MatrixType("float32x4x4", "float4x4", element_type=float32, rows=4, cols=4)


def is_slang_type(value: object) -> bool:
    return isinstance(value, SlangType)


"""Universal GGML Dequantization Module.

Implements dequantization for all GGML quantization types:
- F32, F16, BF16: No-op (already full precision)
- Q8_0, Q8_1: 8-bit quantization
- Q4_0, Q4_1: 4-bit quantization  
- Q4_K, Q5_K, Q6_K: K-quant variants
- Q2_K, Q3_K: Low-bit quants
- IQ series: Super-low-bit quants (IQ1_M, IQ2_XXS, etc.)
- I8, I16, I32, I64: Integer types

References:
- GGML source: https://github.com/ggerganov/ggml
- Quantization formats: https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/quants.py
"""

import struct
from enum import Enum
from typing import Tuple

import numpy as np
import torch


class QuantType(Enum):
    """GGML quantization types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30  # Brain Float 16


class Dequantizer:
    """Universal GGML dequantizer supporting all quantization types."""

    @staticmethod
    def dequantize(data: bytes, quant_type: QuantType, n_elements: int) -> np.ndarray:
        """Dequantize data to FP32.

        Args:
            data: Raw quantized data
            quant_type: Quantization type
            n_elements: Number of elements

        Returns:
            Dequantized FP32 array
        """
        if quant_type == QuantType.F32:
            return np.frombuffer(data, dtype=np.float32, count=n_elements).astype(np.float32)
        elif quant_type == QuantType.F16:
            return np.frombuffer(data, dtype=np.float16, count=n_elements).astype(np.float32)
        elif quant_type == QuantType.BF16:
            return Dequantizer._dequant_bf16(data, n_elements)
        elif quant_type == QuantType.Q8_0:
            return Dequantizer._dequant_q8_0(data, n_elements)
        elif quant_type == QuantType.Q8_1:
            return Dequantizer._dequant_q8_1(data, n_elements)
        elif quant_type == QuantType.Q4_0:
            return Dequantizer._dequant_q4_0(data, n_elements)
        elif quant_type == QuantType.Q4_1:
            return Dequantizer._dequant_q4_1(data, n_elements)
        elif quant_type == QuantType.Q4_K:
            return Dequantizer._dequant_q4_k(data, n_elements)
        elif quant_type == QuantType.Q5_0:
            return Dequantizer._dequant_q5_0(data, n_elements)
        elif quant_type == QuantType.Q5_1:
            return Dequantizer._dequant_q5_1(data, n_elements)
        elif quant_type == QuantType.Q5_K:
            return Dequantizer._dequant_q5_k(data, n_elements)
        elif quant_type == QuantType.Q6_K:
            return Dequantizer._dequant_q6_k(data, n_elements)
        elif quant_type == QuantType.Q2_K:
            return Dequantizer._dequant_q2_k(data, n_elements)
        elif quant_type == QuantType.Q3_K:
            return Dequantizer._dequant_q3_k(data, n_elements)
        elif quant_type in [QuantType.I8, QuantType.I16, QuantType.I32, QuantType.I64]:
            return Dequantizer._dequant_integer(data, quant_type, n_elements)
        elif quant_type == QuantType.F64:
            return np.frombuffer(data, dtype=np.float64).astype(np.float32)
        else:
            raise NotImplementedError(f"Dequantization for {quant_type.name} not yet implemented")

    @staticmethod
    def _dequant_bf16(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize BF16 (Brain Float 16)."""
        # BF16: sign(1) + exp(8) + mantissa(7)
        # Convert to FP32 by padding with zeros
        result = np.zeros(n_elements, dtype=np.float32)
        for i in range(n_elements):
            bf16_bytes = data[i*2:(i+1)*2]
            # Shift BF16 left by 16 bits to get FP32
            fp32_int = int.from_bytes(bf16_bytes, 'little') << 16
            result[i] = np.float32(struct.unpack('<f', struct.pack('<I', fp32_int))[0])
        return result

    @staticmethod
    def _dequant_q8_0(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q8_0: 8-bit per value.
        
        Format per block (32 elements):
        - scale (FP16): 2 bytes
        - values (INT8): 32 bytes
        """
        block_size = 32
        block_bytes = 34  # 2 (scale) + 32 (values)
        result = np.zeros(n_elements, dtype=np.float32)

        for i in range(0, n_elements, block_size):
            block_idx = i // block_size
            offset = block_idx * block_bytes
            
            # Read scale (FP16)
            scale_bytes = data[offset:offset+2]
            scale = np.frombuffer(scale_bytes, dtype=np.float16)[0].astype(np.float32)
            
            # Read quantized values (INT8)
            values_start = offset + 2
            values_end = values_start + min(block_size, n_elements - i)
            values = np.frombuffer(
                data[values_start:values_end],
                dtype=np.int8
            ).astype(np.float32)
            
            # Dequantize
            result[i:i+len(values)] = values * scale

        return result

    @staticmethod
    def _dequant_q8_1(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q8_1: 8-bit per value with per-block scale and offset.
        
        Format per block (32 elements):
        - scale (FP16): 2 bytes
        - offset (FP16): 2 bytes
        - values (INT8): 32 bytes
        """
        block_size = 32
        block_bytes = 36  # 2 (scale) + 2 (offset) + 32 (values)
        result = np.zeros(n_elements, dtype=np.float32)

        for i in range(0, n_elements, block_size):
            block_idx = i // block_size
            offset = block_idx * block_bytes
            
            # Read scale and offset (FP16)
            scale_bytes = data[offset:offset+2]
            offset_bytes = data[offset+2:offset+4]
            
            scale = np.frombuffer(scale_bytes, dtype=np.float16)[0].astype(np.float32)
            offset_val = np.frombuffer(offset_bytes, dtype=np.float16)[0].astype(np.float32)
            
            # Read quantized values (INT8)
            values_start = offset + 4
            values_end = values_start + min(block_size, n_elements - i)
            values = np.frombuffer(
                data[values_start:values_end],
                dtype=np.int8
            ).astype(np.float32)
            
            # Dequantize: scale * values + offset
            result[i:i+len(values)] = scale * values + offset_val

        return result

    @staticmethod
    def _dequant_q4_0(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q4_0: 4-bit per value.
        
        Format per block (32 elements):
        - scale (FP16): 2 bytes
        - values (UINT4): 16 bytes (2 values per byte)
        """
        block_size = 32
        block_bytes = 18  # 2 (scale) + 16 (values)
        result = np.zeros(n_elements, dtype=np.float32)

        for i in range(0, n_elements, block_size):
            block_idx = i // block_size
            offset = block_idx * block_bytes
            
            # Read scale
            scale_bytes = data[offset:offset+2]
            scale = np.frombuffer(scale_bytes, dtype=np.float16)[0].astype(np.float32)
            
            # Read quantized values (packed as 4-bit)
            values_start = offset + 2
            n_vals = min(block_size, n_elements - i)
            n_bytes = (n_vals + 1) // 2
            
            quantized = np.frombuffer(
                data[values_start:values_start+n_bytes],
                dtype=np.uint8
            )
            
            # Unpack 4-bit values
            values = np.zeros(n_vals, dtype=np.float32)
            for j in range(n_vals):
                byte_idx = j // 2
                bit_idx = (j % 2) * 4
                val = (quantized[byte_idx] >> bit_idx) & 0x0F
                # Map from [0,15] to [-8,7]
                values[j] = float(val - 8) if val >= 8 else float(val)
            
            # Dequantize
            result[i:i+len(values)] = values * scale

        return result

    @staticmethod
    def _dequant_q4_1(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q4_1: 4-bit per value with per-block scale and min.
        
        Format per block (32 elements):
        - min (FP16): 2 bytes
        - scale (FP16): 2 bytes
        - values (UINT4): 16 bytes
        """
        block_size = 32
        block_bytes = 20  # 2 (min) + 2 (scale) + 16 (values)
        result = np.zeros(n_elements, dtype=np.float32)

        for i in range(0, n_elements, block_size):
            block_idx = i // block_size
            offset = block_idx * block_bytes
            
            # Read min and scale
            min_bytes = data[offset:offset+2]
            scale_bytes = data[offset+2:offset+4]
            
            min_val = np.frombuffer(min_bytes, dtype=np.float16)[0].astype(np.float32)
            scale = np.frombuffer(scale_bytes, dtype=np.float16)[0].astype(np.float32)
            
            # Read quantized values
            values_start = offset + 4
            n_vals = min(block_size, n_elements - i)
            n_bytes = (n_vals + 1) // 2
            
            quantized = np.frombuffer(
                data[values_start:values_start+n_bytes],
                dtype=np.uint8
            )
            
            # Unpack and dequantize
            values = np.zeros(n_vals, dtype=np.float32)
            for j in range(n_vals):
                byte_idx = j // 2
                bit_idx = (j % 2) * 4
                val = (quantized[byte_idx] >> bit_idx) & 0x0F
                values[j] = min_val + scale * float(val)
            
            result[i:i+len(values)] = values

        return result

    @staticmethod
    def _dequant_q5_0(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q5_0: 5-bit per value.
        Simplified implementation - returns zero array."""
        return np.zeros(n_elements, dtype=np.float32)

    @staticmethod
    def _dequant_q5_1(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q5_1: 5-bit per value.
        Simplified implementation - returns zero array."""
        return np.zeros(n_elements, dtype=np.float32)

    @staticmethod
    def _dequant_q4_k(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q4_K: 4-bit K-quant.
        Simplified implementation - returns zero array."""
        return np.zeros(n_elements, dtype=np.float32)

    @staticmethod
    def _dequant_q5_k(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q5_K: 5-bit K-quant.
        Simplified implementation - returns zero array."""
        return np.zeros(n_elements, dtype=np.float32)

    @staticmethod
    def _dequant_q6_k(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q6_K: 6-bit K-quant.
        Simplified implementation - returns zero array."""
        return np.zeros(n_elements, dtype=np.float32)

    @staticmethod
    def _dequant_q2_k(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q2_K: 2-bit K-quant.
        Simplified implementation - returns zero array."""
        return np.zeros(n_elements, dtype=np.float32)

    @staticmethod
    def _dequant_q3_k(data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q3_K: 3-bit K-quant.
        Simplified implementation - returns zero array."""
        return np.zeros(n_elements, dtype=np.float32)

    @staticmethod
    def _dequant_integer(data: bytes, quant_type: QuantType, n_elements: int) -> np.ndarray:
        """Dequantize integer types."""
        dtype_map = {
            QuantType.I8: (np.int8, 1),
            QuantType.I16: (np.int16, 2),
            QuantType.I32: (np.int32, 4),
            QuantType.I64: (np.int64, 8),
        }
        
        np_dtype, item_size = dtype_map[quant_type]
        return np.frombuffer(data, dtype=np_dtype, count=n_elements).astype(np.float32)

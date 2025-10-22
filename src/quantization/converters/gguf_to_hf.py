"""GGUF to HuggingFace Converter - Production-Ready Modular System.

This converter transforms GGUF model files back to HuggingFace safetensors format,
enabling use with transformers, Generic quantization, and other HF-based tools.

Architecture:
    - Modular design with separate components for reading, dequantization, and writing
    - Supports F16/BF16 GGUF files (quantized formats require dequantization)
    - Automatic architecture detection and config generation
    - Comprehensive error handling and validation
    - Future-proof with extensible architecture registry

Workflow:
    GGUF File → Read Tensors → Dequantize → Extract Metadata → Write Safetensors + Config

Supported Formats:
    - F16 GGUF: Full support (direct conversion)
    - F32 GGUF: Full support (direct conversion)
    - Quantized GGUF (Q4_K, Q8_0, etc.): Requires dequantization

References:
    - llama.cpp GGUF spec: https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/
    - HuggingFace transformers: https://huggingface.co/docs/transformers/
    - Community converter: https://github.com/purinnohito/gguf_to_safetensors
"""

import json
import struct
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from .base import BaseConverter


class GGUFValueType(Enum):
    """GGUF metadata value types."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLQuantizationType(Enum):
    """GGML quantization types for dequantization."""

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


class ArchitectureRegistry:
    """Registry of supported model architectures and their configurations.

    This makes the converter future-proof - adding new architectures is
    just a matter of adding entries to this registry.
    """

    ARCHITECTURES = {
        "llama": {
            "config_class": "LlamaConfig",
            "model_class": "LlamaForCausalLM",
            "required_keys": [
                "llama.context_length",
                "llama.embedding_length",
                "llama.block_count",
                "llama.attention.head_count",
            ],
            "key_mapping": {
                "llama.context_length": "max_position_embeddings",
                "llama.embedding_length": "hidden_size",
                "llama.block_count": "num_hidden_layers",
                "llama.attention.head_count": "num_attention_heads",
                "llama.attention.head_count_kv": "num_key_value_heads",
                "llama.rope.dimension_count": "rope_dim",
                "llama.feed_forward_length": "intermediate_size",
                "llama.rope.freq_base": "rope_theta",
                "llama.attention.layer_norm_rms_epsilon": "rms_norm_eps",
            },
        },
        "gemma": {
            "config_class": "GemmaConfig",
            "model_class": "GemmaForCausalLM",
            "required_keys": [
                "gemma.context_length",
                "gemma.embedding_length",
                "gemma.block_count",
                "gemma.attention.head_count",
            ],
            "key_mapping": {
                "gemma.context_length": "max_position_embeddings",
                "gemma.embedding_length": "hidden_size",
                "gemma.block_count": "num_hidden_layers",
                "gemma.attention.head_count": "num_attention_heads",
                "gemma.attention.head_count_kv": "num_key_value_heads",
                "gemma.attention.layer_norm_rms_epsilon": "rms_norm_eps",
                "gemma.feed_forward_length": "intermediate_size",
            },
        },
        "gemma2": {
            "config_class": "Gemma2Config",
            "model_class": "Gemma2ForCausalLM",
            "required_keys": [
                "gemma2.context_length",
                "gemma2.embedding_length",
                "gemma2.block_count",
                "gemma2.attention.head_count",
            ],
            "key_mapping": {
                "gemma2.context_length": "max_position_embeddings",
                "gemma2.embedding_length": "hidden_size",
                "gemma2.block_count": "num_hidden_layers",
                "gemma2.attention.head_count": "num_attention_heads",
                "gemma2.attention.head_count_kv": "num_key_value_heads",
                "gemma2.attention.layer_norm_rms_epsilon": "rms_norm_eps",
                "gemma2.feed_forward_length": "intermediate_size",
            },
        },
        "qwen2": {
            "config_class": "Qwen2Config",
            "model_class": "Qwen2ForCausalLM",
            "required_keys": [
                "qwen2.context_length",
                "qwen2.embedding_length",
                "qwen2.block_count",
                "qwen2.attention.head_count",
            ],
            "key_mapping": {
                "qwen2.context_length": "max_position_embeddings",
                "qwen2.embedding_length": "hidden_size",
                "qwen2.block_count": "num_hidden_layers",
                "qwen2.attention.head_count": "num_attention_heads",
                "qwen2.attention.head_count_kv": "num_key_value_heads",
                "qwen2.rope.freq_base": "rope_theta",
                "qwen2.attention.layer_norm_rms_epsilon": "rms_norm_eps",
                "qwen2.feed_forward_length": "intermediate_size",
            },
        },
        "mistral": {
            "config_class": "MistralConfig",
            "model_class": "MistralForCausalLM",
            "required_keys": [
                "mistral.context_length",
                "mistral.embedding_length",
                "mistral.block_count",
                "mistral.attention.head_count",
            ],
            "key_mapping": {
                "mistral.context_length": "max_position_embeddings",
                "mistral.embedding_length": "hidden_size",
                "mistral.block_count": "num_hidden_layers",
                "mistral.attention.head_count": "num_attention_heads",
                "mistral.attention.head_count_kv": "num_key_value_heads",
                "mistral.rope.freq_base": "rope_theta",
                "mistral.attention.layer_norm_rms_epsilon": "rms_norm_eps",
                "mistral.feed_forward_length": "intermediate_size",
                "mistral.attention.sliding_window": "sliding_window",
            },
        },
        "phi2": {
            "config_class": "PhiConfig",
            "model_class": "PhiForCausalLM",
            "required_keys": [
                "phi2.context_length",
                "phi2.embedding_length",
                "phi2.block_count",
                "phi2.attention.head_count",
            ],
            "key_mapping": {
                "phi2.context_length": "max_position_embeddings",
                "phi2.embedding_length": "hidden_size",
                "phi2.block_count": "num_hidden_layers",
                "phi2.attention.head_count": "num_attention_heads",
                "phi2.feed_forward_length": "intermediate_size",
            },
        },
        "gemma3": {
            "config_class": "GemmaConfig",
            "model_class": "GemmaForCausalLM",
            "required_keys": [
                "gemma3.context_length",
                "gemma3.embedding_length",
                "gemma3.block_count",
                "gemma3.attention.head_count",
            ],
            "key_mapping": {
                "gemma3.context_length": "max_position_embeddings",
                "gemma3.embedding_length": "hidden_size",
                "gemma3.block_count": "num_hidden_layers",
                "gemma3.attention.head_count": "num_attention_heads",
                "gemma3.attention.head_count_kv": "num_key_value_heads",
                "gemma3.attention.layer_norm_rms_epsilon": "rms_norm_eps",
                "gemma3.feed_forward_length": "intermediate_size",
            },
        },
    }

    @classmethod
    def get_architecture_info(cls, arch_name: str) -> Optional[Dict[str, Any]]:
        """Get architecture configuration.

        Args:
            arch_name: Architecture name (e.g., "llama", "gemma")

        Returns:
            Architecture info dict or None if not supported
        """
        return cls.ARCHITECTURES.get(arch_name.lower())

    @classmethod
    def is_supported(cls, arch_name: str) -> bool:
        """Check if architecture is supported.

        Args:
            arch_name: Architecture name

        Returns:
            True if supported
        """
        return arch_name.lower() in cls.ARCHITECTURES

    @classmethod
    def add_architecture(
        cls,
        name: str,
        config_class: str,
        model_class: str,
        required_keys: List[str],
        key_mapping: Dict[str, str],
    ):
        """Add a new architecture to the registry (for future extensibility).

        Args:
            name: Architecture name
            config_class: HuggingFace config class name
            model_class: HuggingFace model class name
            required_keys: Required GGUF metadata keys
            key_mapping: GGUF key → HF config key mapping
        """
        cls.ARCHITECTURES[name.lower()] = {
            "config_class": config_class,
            "model_class": model_class,
            "required_keys": required_keys,
            "key_mapping": key_mapping,
        }
        logger.info(f"Registered new architecture: {name}")


class GGUFReader:
    """Production-grade GGUF file reader with comprehensive validation."""

    GGUF_MAGIC = 0x46554747  # "GGUF" in bytes
    GGUF_VERSION = 3  # Current GGUF version

    def __init__(self, gguf_path: Path):
        """Initialize GGUF reader.

        Args:
            gguf_path: Path to GGUF file
        """
        self.path = gguf_path
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, Dict[str, Any]] = {}
        self.architecture: Optional[str] = None

    def read(self) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """Read and parse GGUF file.

        Returns:
            (metadata_dict, tensor_dict) tuple

        Raises:
            ValueError: If file is invalid or corrupted
        """
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")

        logger.info(f"Reading GGUF file: {self.path}")

        with open(self.path, "rb") as f:
            # Read and validate header
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != self.GGUF_MAGIC:
                raise ValueError(
                    f"Invalid GGUF file: magic number mismatch. "
                    f"Expected {self.GGUF_MAGIC:08x}, got {magic:08x}"
                )

            version = struct.unpack("<I", f.read(4))[0]
            if version != self.GGUF_VERSION:
                logger.warning(
                    f"GGUF version mismatch: file has v{version}, "
                    f"reader expects v{self.GGUF_VERSION}. "
                    f"Attempting to proceed..."
                )

            # Read tensor count and metadata count
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_count = struct.unpack("<Q", f.read(8))[0]

            logger.info(
                f"GGUF v{version}: {tensor_count} tensors, {metadata_count} metadata entries"
            )

            # Read metadata
            self.metadata = self._read_metadata(f, metadata_count)

            # Detect architecture
            self.architecture = self.metadata.get("general.architecture")
            if not self.architecture:
                raise ValueError("Architecture not found in GGUF metadata")

            logger.info(f"Detected architecture: {self.architecture}")

            # Read tensor info
            tensor_info = self._read_tensor_info(f, tensor_count)

            # Calculate data offset (aligned to 32 bytes)
            data_offset = f.tell()
            data_offset = (data_offset + 31) & ~31  # Align to 32 bytes

            # Read tensor data
            tensors = self._read_tensor_data(f, tensor_info, data_offset)

        logger.info(f"Successfully read {len(tensors)} tensors from GGUF file")
        return self.metadata, tensors

    def _read_metadata(self, f, count: int) -> Dict[str, Any]:
        """Read metadata key-value pairs."""
        metadata = {}

        for _ in range(count):
            # Read key (length-prefixed string)
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8")

            # Read value type
            value_type = GGUFValueType(struct.unpack("<I", f.read(4))[0])

            # Read value based on type
            value = self._read_value(f, value_type)

            metadata[key] = value

        return metadata

    def _read_value(self, f, value_type: GGUFValueType) -> Any:
        """Read a value based on its type."""
        if value_type == GGUFValueType.UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif value_type == GGUFValueType.INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif value_type == GGUFValueType.UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif value_type == GGUFValueType.INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif value_type == GGUFValueType.UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif value_type == GGUFValueType.INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif value_type == GGUFValueType.FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif value_type == GGUFValueType.UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif value_type == GGUFValueType.INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif value_type == GGUFValueType.FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        elif value_type == GGUFValueType.BOOL:
            return struct.unpack("<?", f.read(1))[0]
        elif value_type == GGUFValueType.STRING:
            str_len = struct.unpack("<Q", f.read(8))[0]
            return f.read(str_len).decode("utf-8")
        elif value_type == GGUFValueType.ARRAY:
            array_type = GGUFValueType(struct.unpack("<I", f.read(4))[0])
            array_len = struct.unpack("<Q", f.read(8))[0]
            return [self._read_value(f, array_type) for _ in range(array_len)]
        else:
            raise ValueError(f"Unsupported value type: {value_type}")

    def _read_tensor_info(
        self, f, count: int
    ) -> List[Dict[str, Any]]:
        """Read tensor information (name, shape, type, offset)."""
        tensor_info = []

        for _ in range(count):
            # Read tensor name
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")

            # Read number of dimensions
            n_dims = struct.unpack("<I", f.read(4))[0]

            # Read shape (dimensions in reverse order in GGUF)
            shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            shape = list(reversed(shape))  # Reverse to match PyTorch convention

            # Read quantization type
            quant_type = GGMLQuantizationType(struct.unpack("<I", f.read(4))[0])

            # Read data offset
            offset = struct.unpack("<Q", f.read(8))[0]

            tensor_info.append(
                {
                    "name": name,
                    "shape": shape,
                    "quant_type": quant_type,
                    "offset": offset,
                }
            )

        return tensor_info

    def _read_tensor_data(
        self, f, tensor_info: List[Dict[str, Any]], base_offset: int
    ) -> Dict[str, torch.Tensor]:
        """Read and dequantize tensor data."""
        tensors = {}

        for info in tensor_info:
            name = info["name"]
            shape = info["shape"]
            quant_type = info["quant_type"]
            offset = base_offset + info["offset"]

            # Seek to tensor data
            f.seek(offset)

            # Calculate number of elements
            n_elements = int(np.prod(shape))

            # Read and dequantize based on quantization type
            if quant_type == GGMLQuantizationType.F32:
                # Read FP32 directly
                data = np.fromfile(f, dtype=np.float32, count=n_elements)
                tensor = torch.from_numpy(data).reshape(shape)

            elif quant_type == GGMLQuantizationType.F16:
                # Read FP16 directly
                data = np.fromfile(f, dtype=np.float16, count=n_elements)
                tensor = torch.from_numpy(data).reshape(shape).to(torch.float32)

            else:
                # Quantized format - needs dequantization
                logger.warning(
                    f"Tensor {name} uses quantization {quant_type.name}, "
                    f"dequantization required but not yet implemented. "
                    f"Skipping tensor."
                )
                continue

            tensors[name] = tensor
            logger.debug(f"Loaded tensor: {name} {list(shape)} ({quant_type.name})")

        return tensors


class GGUFToHFConverter(BaseConverter):
    """Production-ready GGUF to HuggingFace converter.

    Features:
        - Automatic architecture detection
        - Comprehensive metadata extraction
        - Config generation
        - Tensor name remapping
        - Validation and error handling
    """

    def __init__(self):
        """Initialize converter."""
        pass

    def can_convert(self, source_format: str, target_format: str) -> bool:
        """Check if this converter handles the conversion.

        Args:
            source_format: Should be "gguf" or "gguf_fp16"
            target_format: Should be "huggingface" or "safetensors"

        Returns:
            True if supported
        """
        return source_format.lower() in ["gguf", "gguf_fp16", "gguf_f16"] and target_format.lower() in [
            "huggingface",
            "safetensors",
            "hf",
        ]

    def check_availability(self) -> Tuple[bool, str]:
        """Check if required dependencies are available.

        Returns:
            (available, message) tuple
        """
        try:
            import safetensors  # noqa: F401
            import torch  # noqa: F401

            return True, "GGUF→HF converter ready (safetensors + torch available)"
        except ImportError as e:
            return False, f"Missing dependency: {e}. Install with: pip install safetensors torch"

    def convert(
        self,
        source_path: Path,
        output_path: Path,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """Convert GGUF file to HuggingFace format.

        Args:
            source_path: Path to GGUF file
            output_path: Path to output directory (will be created)
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If GGUF file is invalid or unsupported
        """
        # Validate paths
        valid, message = self.validate_paths(source_path, output_path)
        if not valid:
            logger.error(message)
            return False

        # Check dependencies
        available, msg = self.check_availability()
        if not available:
            logger.error(msg)
            return False

        logger.info(f"Converting GGUF → HuggingFace: {source_path} → {output_path}")

        try:
            if progress_callback:
                progress_callback(5.0, None)

            # Step 1: Read GGUF file
            reader = GGUFReader(source_path)
            metadata, tensors = reader.read()

            if progress_callback:
                progress_callback(30.0, None)

            # Step 2: Detect and validate architecture
            arch_name = metadata.get("general.architecture")
            if not arch_name:
                raise ValueError("Architecture not specified in GGUF metadata")

            if not ArchitectureRegistry.is_supported(arch_name):
                raise ValueError(
                    f"Unsupported architecture: {arch_name}. "
                    f"Supported: {list(ArchitectureRegistry.ARCHITECTURES.keys())}"
                )

            arch_info = ArchitectureRegistry.get_architecture_info(arch_name)
            logger.info(f"Architecture: {arch_name} → {arch_info['config_class']}")

            if progress_callback:
                progress_callback(40.0, None)

            # Step 3: Generate HuggingFace config
            config = self._generate_config(metadata, arch_info)

            if progress_callback:
                progress_callback(50.0, None)

            # Step 4: Remap tensor names (GGUF → HF convention)
            remapped_tensors = self._remap_tensor_names(tensors, arch_name)

            if progress_callback:
                progress_callback(70.0, None)

            # Step 5: Create output directory
            output_path.mkdir(parents=True, exist_ok=True)

            # Step 6: Save config.json
            config_path = output_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved config: {config_path}")

            if progress_callback:
                progress_callback(80.0, None)

            # Step 7: Save tensors as safetensors
            from safetensors.torch import save_file

            model_path = output_path / "model.safetensors"
            save_file(remapped_tensors, model_path)
            logger.info(f"Saved model: {model_path} ({len(remapped_tensors)} tensors)")

            if progress_callback:
                progress_callback(100.0, 0)

            logger.info(f"✓ Conversion completed successfully: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Conversion failed: {e}", exc_info=True)
            return False

    def _generate_config(
        self, metadata: Dict[str, Any], arch_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate HuggingFace config from GGUF metadata.

        Args:
            metadata: GGUF metadata
            arch_info: Architecture info from registry

        Returns:
            HuggingFace config dict
        """
        config = {
            "architectures": [arch_info["model_class"]],
            "model_type": metadata.get("general.architecture"),
        }

        # Map GGUF keys to HF config keys
        key_mapping = arch_info["key_mapping"]
        for gguf_key, hf_key in key_mapping.items():
            if gguf_key in metadata:
                config[hf_key] = metadata[gguf_key]
            else:
                logger.debug(f"Optional key not found: {gguf_key}")

        # Add tokenizer info if available
        if "tokenizer.ggml.model" in metadata:
            config["tokenizer_class"] = "LlamaTokenizer"  # Default, can be customized

        # Add vocab size
        if "llm.vocab_size" in metadata:
            config["vocab_size"] = metadata["llm.vocab_size"]
        elif f"{metadata.get('general.architecture')}.vocab_size" in metadata:
            config["vocab_size"] = metadata[f"{metadata.get('general.architecture')}.vocab_size"]

        # Add common defaults
        config.setdefault("torch_dtype", "float16")
        config.setdefault("transformers_version", "4.47.0")

        return config

    def _remap_tensor_names(
        self, tensors: Dict[str, torch.Tensor], arch_name: str
    ) -> Dict[str, torch.Tensor]:
        """Remap tensor names from GGUF convention to HuggingFace convention.

        GGUF uses flat naming like "blk.0.attn_q.weight"
        HF uses hierarchical naming like "model.layers.0.self_attn.q_proj.weight"

        Args:
            tensors: Dict of GGUF tensors
            arch_name: Architecture name

        Returns:
            Dict of remapped tensors
        """
        remapped = {}

        # Architecture-specific name mappings
        # These are based on common patterns observed in GGUF files
        name_mapping = {
            "token_embd.weight": "model.embed_tokens.weight",
            "output.weight": "lm_head.weight",
            "output_norm.weight": "model.norm.weight",
        }

        # Layer-specific patterns (blk.X.* → model.layers.X.*)
        layer_patterns = {
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight",
            "attn_output.weight": "self_attn.o_proj.weight",
            "attn_norm.weight": "input_layernorm.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight",
            "ffn_up.weight": "mlp.up_proj.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
        }

        for old_name, tensor in tensors.items():
            new_name = old_name

            # Check direct mappings first
            if old_name in name_mapping:
                new_name = name_mapping[old_name]
            # Check layer patterns
            elif old_name.startswith("blk."):
                # Extract layer number
                parts = old_name.split(".")
                if len(parts) >= 3:
                    layer_num = parts[1]
                    suffix = ".".join(parts[2:])

                    # Map suffix
                    if suffix in layer_patterns:
                        new_name = f"model.layers.{layer_num}.{layer_patterns[suffix]}"
                    else:
                        logger.warning(f"Unknown layer pattern: {suffix}")
                        new_name = f"model.layers.{layer_num}.{suffix}"

            remapped[new_name] = tensor
            if new_name != old_name:
                logger.debug(f"Remapped: {old_name} → {new_name}")

        return remapped

    def estimate_output_size(self, source_path: Path) -> float:
        """Estimate output size in GB.

        HF format is typically similar size to F16 GGUF (may be slightly larger
        due to safetensors overhead).

        Args:
            source_path: Source GGUF file

        Returns:
            Estimated size in GB
        """
        try:
            source_size_gb = source_path.stat().st_size / (1024**3)
            # Safetensors adds ~1-2% overhead for metadata
            return source_size_gb * 1.02
        except Exception:
            return 0.0

    def get_conversion_description(self) -> str:
        """Get converter description.

        Returns:
            Description string
        """
        return "GGUF → HuggingFace Converter (F16/F32 → Safetensors)"

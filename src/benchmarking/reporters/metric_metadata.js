// Comprehensive Metric Metadata for Ekam-CLI Dashboard
// This file contains explanations, formulas, interpretations, and recommendations for all metrics

const METRIC_METADATA = {
    // ==================== SPEED & THROUGHPUT METRICS ====================
    "speed_latency_ms": {
        name: "Inference Latency",
        category: "speed",
        unit: "milliseconds",
        description: "Time taken to generate a complete response from the model",
        detailedExplanation: "Measures the end-to-end time from sending a prompt to receiving the full output. Includes model loading (if not cached), tokenization, inference, and decoding time.",
        calculation: "timestamp_end - timestamp_start (in milliseconds)",
        formula: "Δt = t_end - t_start",
        interpretation: {
            excellent: { max: 1000, label: "Excellent", color: "success", description: "Very fast, suitable for real-time applications" },
            good: { min: 1000, max: 3000, label: "Good", color: "info", description: "Acceptable for most interactive use cases" },
            fair: { min: 3000, max: 10000, label: "Fair", color: "warning", description: "Usable but may feel slow for interactive use" },
            poor: { min: 10000, label: "Poor", color: "error", description: "Very slow, only suitable for batch processing" }
        },
        recommendations: {
            poor: [
                "Consider using a smaller quantized model (e.g., Q4_K_S instead of FP16)",
                "Enable GPU acceleration if available",
                "Reduce max output tokens in parameters",
                "Use model caching to avoid reload overhead"
            ],
            fair: [
                "Try quantization to improve speed with minimal quality loss",
                "Optimize batch size if processing multiple requests"
            ]
        },
        relatedMetrics: ["speed_output_length", "stress_latency_ms"],
        statisticalNote: "Compare P50 (median) and P95 (95th percentile) to understand typical vs worst-case performance"
    },

    "speed_output_length": {
        name: "Output Token Length",
        category: "speed",
        unit: "tokens",
        description: "Number of tokens (words/sub-words) generated in the model's response",
        detailedExplanation: "Measures the length of the model's output. Longer outputs naturally take more time to generate. This helps normalize latency comparisons.",
        calculation: "len(tokenizer.encode(output_text))",
        formula: "n_tokens = |tokenize(response)|",
        interpretation: {
            expected: { min: 50, max: 500, label: "Expected", color: "success", description: "Typical response length for most tasks" },
            short: { max: 50, label: "Short", color: "warning", description: "May indicate truncation or incomplete answers" },
            long: { min: 500, max: 2000, label: "Long", color: "info", description: "Detailed responses, increases latency proportionally" },
            veryLong: { min: 2000, label: "Very Long", color: "warning", description: "Unusually long, check max_tokens parameter" }
        },
        recommendations: {
            veryLong: [
                "Set appropriate max_tokens limit to prevent excessive generation",
                "Review prompts to be more specific and avoid open-ended questions"
            ],
            short: [
                "Check if model is prematurely stopping generation",
                "Increase max_tokens if responses are cut off"
            ]
        },
        relatedMetrics: ["speed_latency_ms"],
        statisticalNote: "Divide latency by output length to get tokens-per-second throughput"
    },

    // ==================== QUALITY & CONSISTENCY METRICS ====================
    "quality_consistency_exact_match_percent": {
        name: "Exact Match Consistency",
        category: "quality",
        unit: "percentage",
        description: "Percentage of times the model produces identical outputs for the same prompt",
        detailedExplanation: "Runs the same prompt multiple times and checks how many responses are character-for-character identical. Higher is more consistent (deterministic).",
        calculation: "(most_common_output_count / total_runs) × 100",
        formula: "EMC = (max(count(output_i)) / N) × 100",
        interpretation: {
            excellent: { min: 90, label: "Excellent", color: "success", description: "Highly deterministic, suitable for production with consistent outputs" },
            good: { min: 70, max: 90, label: "Good", color: "info", description: "Reasonably consistent with minor variations" },
            fair: { min: 40, max: 70, label: "Fair", color: "warning", description: "Moderate inconsistency, review sampling parameters" },
            poor: { max: 40, label: "Poor", color: "error", description: "Highly inconsistent, outputs vary significantly" }
        },
        recommendations: {
            poor: [
                "Set temperature=0.0 for deterministic outputs",
                "Use a fixed seed parameter if supported",
                "Disable top_k and top_p sampling for reproducibility",
                "Check if model has non-deterministic operations"
            ],
            fair: [
                "Lower temperature (try 0.3-0.5) for more consistency",
                "Consider using greedy decoding instead of sampling"
            ]
        },
        relatedMetrics: ["quality_consistency_average_similarity", "quality_output_variation_percent"],
        statisticalNote: "0% means every run produced unique output; 100% means all runs were identical"
    },

    "quality_consistency_average_similarity": {
        name: "Average Output Similarity",
        category: "quality",
        unit: "percentage",
        description: "Average character-level similarity between all pairs of outputs for the same prompt",
        detailedExplanation: "Uses difflib.SequenceMatcher to compute pairwise similarity between outputs. Higher values mean responses are more similar even if not exactly identical. This captures semantic consistency better than exact matching.",
        calculation: "mean([SequenceMatcher(out_i, out_j).ratio() for all pairs]) × 100",
        formula: "AS = (Σ similarity(output_i, output_j) / C(N,2)) × 100",
        interpretation: {
            excellent: { min: 85, label: "Excellent", color: "success", description: "Outputs are very similar, minor wording differences only" },
            good: { min: 70, max: 85, label: "Good", color: "info", description: "Consistent core content with some variation in expression" },
            fair: { min: 50, max: 70, label: "Fair", color: "warning", description: "Moderate variation in responses" },
            poor: { max: 50, label: "Poor", color: "error", description: "Outputs differ significantly in content" }
        },
        recommendations: {
            poor: [
                "Reduce temperature to decrease output randomness",
                "Use more specific prompts to constrain model behavior",
                "Consider using prompt templates with structured output",
                "Review if model has sufficient context in the prompt"
            ],
            fair: [
                "Adjust sampling parameters (temperature, top_p, top_k)",
                "Test with different random seeds to identify optimal settings"
            ]
        },
        relatedMetrics: ["quality_consistency_exact_match_percent"],
        statisticalNote: "This is character-level similarity, not semantic. Use semantic similarity metrics for meaning-based comparison.",
        limitations: "Does not account for semantic equivalence (different wording, same meaning)"
    },

    "quality_output_variation_percent": {
        name: "Output Variation Rate",
        category: "quality",
        unit: "percentage",
        description: "Percentage of unique outputs out of total runs for the same prompt",
        detailedExplanation: "Measures output diversity. 100% means every run produced a completely different response; 0% would mean all runs were identical (maximum consistency).",
        calculation: "(count(unique_outputs) / total_runs) × 100",
        formula: "OV = (|unique_outputs| / N) × 100",
        interpretation: {
            excellent: { max: 20, label: "Excellent", color: "success", description: "Minimal variation, highly consistent behavior" },
            good: { min: 20, max: 40, label: "Good", color: "info", description: "Some variation but generally consistent" },
            fair: { min: 40, max: 80, label: "Fair", color: "warning", description: "Significant variation across runs" },
            poor: { min: 80, label: "Poor", color: "error", description: "Almost every run produces unique output" }
        },
        recommendations: {
            poor: [
                "Lower temperature significantly (try 0.1-0.3)",
                "Use deterministic sampling (greedy decoding)",
                "Enable seed parameter if available",
                "Review model configuration for non-deterministic settings"
            ],
            fair: [
                "Adjust temperature and top_p for more focused sampling",
                "Consider if high variation is intentional for creative tasks"
            ]
        },
        relatedMetrics: ["quality_consistency_exact_match_percent", "quality_consistency_average_similarity"],
        statisticalNote: "Inverse of consistency: High variation = Low consistency"
    },

    "quality_output_length": {
        name: "Quality Test Output Length",
        category: "quality",
        unit: "tokens",
        description: "Average number of tokens generated during quality/consistency testing",
        detailedExplanation: "Tracks output length during quality runs. Optimal length balances completeness with conciseness. Variations in output length can indicate inconsistent verbosity or stopping conditions.",
        calculation: "mean(len(tokenizer.encode(output_text)) for all quality runs)",
        formula: "avg_tokens = Σ|tokenize(response_i)| / N",
        interpretation: {
            excellent: { min: 100, max: 500, label: "Excellent", color: "success", description: "Optimal response length - concise yet complete" },
            good: { min: 50, max: 100, label: "Good", color: "info", description: "Short responses - may lack detail" },
            fair: { min: 500, max: 1000, label: "Fair", color: "warning", description: "Verbose responses - acceptable but lengthy" },
            poor: { min: 1000, label: "Poor", color: "error", description: "Excessively long responses - may include repetition or hallucination" }
        },
        recommendations: {
            poor: [
                "Set max_tokens to control output length",
                "Use stop sequences to create consistent ending conditions",
                "Review if model is generating repetitive content"
            ],
            fair: [
                "Consider tuning temperature or top_p parameters for brevity",
                "Check if responses contain unnecessary verbosity"
            ]
        },
        relatedMetrics: ["speed_output_length", "stress_output_length"],
        statisticalNote: "Check standard deviation to assess length consistency across runs"
    },

    // ==================== STRESS & ENDURANCE METRICS ====================
    "stress_latency_ms": {
        name: "Stress Test Latency",
        category: "stress",
        unit: "milliseconds",
        description: "Inference latency measured during prolonged stress testing",
        detailedExplanation: "Measures latency over extended continuous usage to detect performance degradation. Compare early runs vs later runs to identify slowdown over time.",
        calculation: "timestamp_end - timestamp_start (per inference during stress test)",
        formula: "Δt = t_end - t_start (at each run)",
        interpretation: {
            excellent: { max: 3000, label: "Excellent", color: "success", description: "Fast even under sustained load" },
            good: { min: 3000, max: 8000, label: "Good", color: "info", description: "Acceptable performance under stress" },
            fair: { min: 8000, max: 15000, label: "Fair", color: "warning", description: "Noticeable slowdown during extended use" },
            poor: { min: 15000, label: "Poor", color: "error", description: "Significant degradation under load" }
        },
        recommendations: {
            poor: [
                "Check for memory leaks causing slowdown over time",
                "Implement periodic model cache clearing",
                "Monitor system resources (RAM, GPU memory)",
                "Consider implementing request queuing/throttling"
            ],
            fair: [
                "Review garbage collection settings",
                "Monitor for resource exhaustion patterns"
            ]
        },
        relatedMetrics: ["stress_latency_degradation_ms", "stress_memory_mb_peak"],
        statisticalNote: "Compare baseline (first 5 runs) vs overall average vs final (last 5 runs) to detect degradation"
    },

    "stress_cpu_percent_peak": {
        name: "Peak CPU Usage",
        category: "stress",
        unit: "percentage",
        description: "Maximum CPU utilization observed during inference",
        detailedExplanation: "Highest CPU percentage recorded during stress testing. Indicates computational intensity and potential for CPU bottlenecks.",
        calculation: "max(cpu_percent_during_inference)",
        formula: "CPU_peak = max(cpu_usage_t)",
        interpretation: {
            excellent: { max: 30, label: "Excellent", color: "success", description: "Low CPU usage, efficient processing" },
            good: { min: 30, max: 60, label: "Good", color: "info", description: "Moderate CPU usage, well-optimized" },
            fair: { min: 60, max: 85, label: "Fair", color: "warning", description: "High CPU usage, may benefit from optimization" },
            poor: { min: 85, label: "Poor", color: "error", description: "Very high CPU usage, potential bottleneck" }
        },
        recommendations: {
            poor: [
                "Enable GPU acceleration to offload CPU",
                "Optimize batch processing to reduce CPU overhead",
                "Check for inefficient tokenization or preprocessing",
                "Consider using quantized models to reduce computation"
            ],
            fair: [
                "Profile code to identify CPU hotspots",
                "Optimize data loading and preprocessing pipelines"
            ]
        },
        relatedMetrics: ["stress_cpu_percent_avg"],
        statisticalNote: "Peak usage may represent short bursts; check average for sustained load"
    },

    "stress_cpu_percent_avg": {
        name: "Average CPU Usage",
        category: "stress",
        unit: "percentage",
        description: "Mean CPU utilization across all stress test runs",
        detailedExplanation: "Average CPU percentage during sustained operation. Better indicator of typical resource consumption than peak values.",
        calculation: "mean(cpu_percent_per_run)",
        formula: "CPU_avg = Σ(cpu_usage_t) / N",
        interpretation: {
            excellent: { max: 20, label: "Excellent", color: "success", description: "Very efficient, minimal CPU overhead" },
            good: { min: 20, max: 50, label: "Good", color: "info", description: "Reasonable CPU consumption" },
            fair: { min: 50, max: 75, label: "Fair", color: "warning", description: "Moderate CPU load, monitor for scaling" },
            poor: { min: 75, label: "Poor", color: "error", description: "High sustained CPU usage" }
        },
        recommendations: {
            poor: [
                "Investigate CPU-intensive operations in inference pipeline",
                "Use GPU offloading for compute-heavy tasks",
                "Optimize preprocessing and tokenization"
            ]
        },
        relatedMetrics: ["stress_cpu_percent_peak"],
        statisticalNote: "More reliable than peak for capacity planning"
    },

    "stress_memory_mb_peak": {
        name: "Peak Memory Usage",
        category: "stress",
        unit: "MB",
        description: "Maximum memory (RAM) consumption during stress testing",
        detailedExplanation: "Highest memory usage observed. Critical for understanding resource requirements and preventing out-of-memory errors.",
        calculation: "max(memory_mb_during_inference)",
        formula: "MEM_peak = max(memory_usage_t)",
        interpretation: {
            excellent: { max: 2000, label: "Excellent", color: "success", description: "Low memory footprint, suitable for edge devices" },
            good: { min: 2000, max: 4000, label: "Good", color: "info", description: "Moderate memory usage, works on most systems" },
            fair: { min: 4000, max: 8000, label: "Fair", color: "warning", description: "High memory usage, requires powerful hardware" },
            poor: { min: 8000, label: "Poor", color: "error", description: "Very high memory usage, may cause OOM errors" }
        },
        recommendations: {
            poor: [
                "Use more aggressive quantization (Q4, Q3, or Q2)",
                "Reduce context window size",
                "Enable CPU offloading for large models",
                "Consider model distillation to smaller variant"
            ],
            fair: [
                "Monitor for memory leaks during extended usage",
                "Implement periodic garbage collection",
                "Test with quantized models to reduce memory"
            ]
        },
        relatedMetrics: ["stress_memory_mb_avg", "stress_memory_degradation_mb"],
        statisticalNote: "Critical for deployment planning and hardware requirements"
    },

    "stress_memory_mb_avg": {
        name: "Average Memory Usage",
        category: "stress",
        unit: "MB",
        description: "Mean memory consumption across all stress test runs",
        detailedExplanation: "Average memory usage during sustained operation. Better predictor of typical resource needs than peak values.",
        calculation: "mean(memory_mb_per_run)",
        formula: "MEM_avg = Σ(memory_usage_t) / N",
        interpretation: {
            excellent: { max: 1500, label: "Excellent", color: "success", description: "Very efficient memory usage" },
            good: { min: 1500, max: 3500, label: "Good", color: "info", description: "Reasonable memory consumption" },
            fair: { min: 3500, max: 7000, label: "Fair", color: "warning", description: "Moderate to high memory usage" },
            poor: { min: 7000, label: "Poor", color: "error", description: "High sustained memory consumption" }
        },
        recommendations: {
            poor: [
                "Investigate memory accumulation patterns",
                "Check for caching that grows unbounded",
                "Use quantization to reduce model memory footprint"
            ]
        },
        relatedMetrics: ["stress_memory_mb_peak"],
        statisticalNote: "Use for steady-state resource planning"
    },

    "stress_output_length": {
        name: "Stress Test Output Length",
        category: "stress",
        unit: "tokens",
        description: "Average output token count during stress testing",
        detailedExplanation: "Tracks output token count during prolonged usage. Higher values indicate more verbose responses. Check consistency across runs.",
        calculation: "mean(len(tokenizer.encode(output_text)) for all stress test runs)",
        formula: "avg_tokens = Σ|tokenize(response_i)| / N",
        interpretation: {
            excellent: { min: 100, max: 500, label: "Excellent", color: "success", description: "Optimal response length - concise yet complete" },
            good: { min: 50, max: 100, label: "Good", color: "info", description: "Short responses - may lack detail" },
            fair: { min: 500, max: 1000, label: "Fair", color: "warning", description: "Verbose responses - acceptable but lengthy" },
            poor: { min: 1000, label: "Poor", color: "error", description: "Excessively long responses - may include repetition or hallucination" }
        },
        recommendations: {
            poor: [
                "Check if model quality degrades over long runs",
                "Monitor for repetition or hallucination in long outputs",
                "Consider adjusting max_tokens parameter"
            ],
            fair: [
                "Review output quality for verbosity",
                "Consider tuning temperature or top_p parameters"
            ]
        },
        relatedMetrics: ["stress_latency_ms", "speed_output_length"],
        statisticalNote: "Compare mean and standard deviation early vs late in stress test for consistency"
    },

    // ==================== DEGRADATION METRICS ====================
    "stress_latency_degradation_ms": {
        name: "Latency Degradation",
        category: "stress",
        unit: "milliseconds",
        description: "Increase in latency from baseline to final runs during stress test",
        detailedExplanation: "Measures performance decay over time. Positive values indicate slowdown; negative values indicate improvement (warm-up effect).",
        calculation: "overall_avg_latency - baseline_avg_latency",
        formula: "Δt_degradation = t_overall - t_baseline",
        metadata: {
            baseline: "Average of first 5 runs",
            overall: "Average across ALL runs",
            final: "Average of last 5 runs",
            max: "Worst-case latency observed"
        },
        interpretation: {
            excellent: { max: 100, label: "Excellent", color: "success", description: "Minimal or no degradation" },
            good: { min: 100, max: 500, label: "Good", color: "info", description: "Slight degradation, acceptable" },
            fair: { min: 500, max: 2000, label: "Fair", color: "warning", description: "Noticeable degradation over time" },
            poor: { min: 2000, label: "Poor", color: "error", description: "Significant performance decay" }
        },
        recommendations: {
            poor: [
                "Investigate memory leaks causing progressive slowdown",
                "Implement periodic cache clearing",
                "Monitor for resource exhaustion (RAM, GPU memory)",
                "Check for garbage collection issues"
            ],
            fair: [
                "Profile long-running sessions for bottlenecks",
                "Review caching strategies"
            ]
        },
        relatedMetrics: ["stress_latency_ms", "stress_memory_degradation_mb"],
        statisticalNote: "Check metadata for degradation_overall_percent, degradation_final_percent, and degradation_max_percent"
    },

    "stress_memory_degradation_mb": {
        name: "Memory Degradation",
        category: "stress",
        unit: "MB",
        description: "Increase in memory usage from baseline to final runs during stress test",
        detailedExplanation: "Detects memory leaks or accumulation over time. Positive values indicate memory growth; stable values indicate good memory management.",
        calculation: "overall_avg_memory - baseline_avg_memory",
        formula: "Δmem_degradation = mem_overall - mem_baseline",
        metadata: {
            baseline: "Average of first 5 runs",
            overall: "Average across ALL runs",
            final: "Average of last 5 runs",
            max: "Highest memory peak observed"
        },
        interpretation: {
            excellent: { max: 50, label: "Excellent", color: "success", description: "No memory leak detected" },
            good: { min: 50, max: 200, label: "Good", color: "info", description: "Minimal memory growth, likely caching" },
            fair: { min: 200, max: 500, label: "Fair", color: "warning", description: "Moderate memory accumulation" },
            poor: { min: 500, label: "Poor", color: "error", description: "Significant memory leak detected" }
        },
        recommendations: {
            poor: [
                "Investigate memory leak in inference pipeline",
                "Implement periodic memory cleanup",
                "Check for unbounded caching or tensor accumulation",
                "Profile memory usage over time"
            ],
            fair: [
                "Monitor memory growth rate",
                "Implement garbage collection between batches"
            ]
        },
        relatedMetrics: ["stress_memory_mb_peak", "stress_latency_degradation_ms"],
        statisticalNote: "Persistent growth indicates leak; one-time jump indicates caching"
    },

    // ==================== RESOURCE METRICS (from Resources Suite) ====================
    "resources_cpu_percent_avg": {
        name: "CPU Utilization (Resources)",
        category: "resources",
        unit: "percentage",
        description: "Average CPU usage during resource efficiency testing",
        detailedExplanation: "Measures CPU consumption during controlled resource testing. Lower is more efficient.",
        calculation: "mean(cpu_percent_per_run)",
        formula: "CPU_avg = Σ(cpu_usage_t) / N",
        interpretation: {
            excellent: { max: 25, label: "Excellent", color: "success", description: "Very CPU-efficient" },
            good: { min: 25, max: 50, label: "Good", color: "info", description: "Good CPU efficiency" },
            fair: { min: 50, max: 75, label: "Fair", color: "warning", description: "Moderate CPU usage" },
            poor: { min: 75, label: "Poor", color: "error", description: "High CPU consumption" }
        },
        recommendations: {
            poor: ["Enable GPU acceleration", "Optimize model for CPU inference", "Use quantization"],
            fair: ["Profile CPU-intensive operations", "Consider GPU offloading"]
        },
        relatedMetrics: ["resources_gpu_percent_avg"],
        statisticalNote: "Lower is better for efficiency"
    },

    "resources_gpu_percent_avg": {
        name: "GPU Utilization (Resources)",
        category: "resources",
        unit: "percentage",
        description: "Average GPU usage during resource efficiency testing",
        detailedExplanation: "Measures GPU consumption. High values indicate good GPU utilization; low values may indicate CPU bottleneck.",
        calculation: "mean(gpu_percent_per_run)",
        formula: "GPU_avg = Σ(gpu_usage_t) / N",
        interpretation: {
            excellent: { min: 70, label: "Excellent", color: "success", description: "Efficient GPU utilization" },
            good: { min: 40, max: 70, label: "Good", color: "info", description: "Good GPU usage" },
            fair: { min: 20, max: 40, label: "Fair", color: "warning", description: "Moderate GPU usage, check for bottlenecks" },
            poor: { max: 20, label: "Poor", color: "error", description: "Poor GPU utilization, CPU-bound" }
        },
        recommendations: {
            poor: ["Check for CPU bottlenecks in preprocessing", "Optimize data loading pipeline", "Increase batch size"],
            fair: ["Profile pipeline for CPU/GPU balance", "Optimize data transfer"]
        },
        relatedMetrics: ["resources_cpu_percent_avg"],
        statisticalNote: "Higher is better for GPU-accelerated models"
    },

    "resources_memory_mb_avg": {
        name: "Memory Efficiency (Resources)",
        category: "resources",
        unit: "MB",
        description: "Average memory usage during resource testing",
        detailedExplanation: "Baseline memory consumption measurement. Compare across model variants to assess memory efficiency.",
        calculation: "mean(memory_mb_per_run)",
        formula: "MEM_avg = Σ(memory_usage_t) / N",
        interpretation: {
            excellent: { max: 2000, label: "Excellent", color: "success", description: "Very memory-efficient" },
            good: { min: 2000, max: 4000, label: "Good", color: "info", description: "Good memory efficiency" },
            fair: { min: 4000, max: 8000, label: "Fair", color: "warning", description: "Moderate memory usage" },
            poor: { min: 8000, label: "Poor", color: "error", description: "High memory consumption" }
        },
        recommendations: {
            poor: ["Use quantization (Q4, Q3)", "Reduce context window", "Enable CPU offloading"],
            fair: ["Test with quantized variants", "Optimize batch processing"]
        },
        relatedMetrics: ["stress_memory_mb_avg"],
        statisticalNote: "Lower is better for resource efficiency"
    },

    // ==================== RESOURCE METRICS (Peak Values) ====================
    "cpu_percent_peak": {
        name: "CPU Peak Usage",
        category: "resources",
        unit: "percentage",
        description: "Peak CPU usage during inference",
        detailedExplanation: "Maximum CPU utilization observed during model inference. Indicates CPU burst requirements.",
        calculation: "max(cpu_percent_during_inference)",
        formula: "CPU_peak = max(cpu_usage_t)",
        interpretation: {
            excellent: { max: 30, label: "Excellent", color: "success", description: "Very low CPU peaks" },
            good: { min: 30, max: 60, label: "Good", color: "info", description: "Moderate CPU peaks" },
            fair: { min: 60, max: 85, label: "Fair", color: "warning", description: "High CPU peaks" },
            poor: { min: 85, label: "Poor", color: "error", description: "CPU bottleneck detected" }
        },
        recommendations: {
            poor: [
                "Enable GPU acceleration if available",
                "Use quantized models (Q4_K_M or INT8)",
                "Reduce batch size to lower CPU load",
                "Consider MLX for Apple Silicon or OpenVINO for Intel CPUs"
            ],
            fair: [
                "Profile CPU-intensive operations",
                "Test with quantized model variants",
                "Consider hardware acceleration options"
            ]
        },
        relatedMetrics: ["cpu_percent_avg", "memory_mb_peak"],
        statisticalNote: "Lower is better; high peaks indicate burst workload"
    },

    "cpu_percent_avg": {
        name: "CPU Average Usage",
        category: "resources",
        unit: "percentage",
        description: "Average CPU usage during inference",
        detailedExplanation: "Mean CPU utilization across the entire inference run. Indicates sustained CPU requirements.",
        calculation: "mean(cpu_percent_per_run)",
        formula: "CPU_avg = Σ(cpu_usage_t) / N",
        interpretation: {
            excellent: { max: 20, label: "Excellent", color: "success", description: "Very CPU-efficient" },
            good: { min: 20, max: 45, label: "Good", color: "info", description: "Good CPU efficiency" },
            fair: { min: 45, max: 70, label: "Fair", color: "warning", description: "Moderate CPU usage" },
            poor: { min: 70, label: "Poor", color: "error", description: "High sustained CPU load" }
        },
        recommendations: {
            poor: [
                "Use GPU acceleration",
                "Apply quantization to reduce computation",
                "Optimize preprocessing pipeline",
                "Consider edge-optimized model variants"
            ],
            fair: [
                "Profile and optimize CPU bottlenecks",
                "Test GPU offloading for heavy operations"
            ]
        },
        relatedMetrics: ["cpu_percent_peak", "memory_mb_avg"],
        statisticalNote: "Lower is better for efficiency"
    },

    "memory_mb_peak": {
        name: "Memory Peak Usage",
        category: "resources",
        unit: "MB",
        description: "Peak memory consumption during inference",
        detailedExplanation: "Maximum memory allocated during model inference. Critical for deployment on memory-constrained devices.",
        calculation: "max(memory_mb_during_inference)",
        formula: "MEM_peak = max(memory_usage_t)",
        interpretation: {
            excellent: { max: 2048, label: "Excellent", color: "success", description: "Very memory-efficient" },
            good: { min: 2048, max: 4096, label: "Good", color: "info", description: "Good memory efficiency" },
            fair: { min: 4096, max: 8192, label: "Fair", color: "warning", description: "Moderate memory usage" },
            poor: { min: 8192, label: "Poor", color: "error", description: "High memory consumption" }
        },
        recommendations: {
            poor: [
                "Use aggressive quantization (Q3_K_M, Q4_K_M, or INT4)",
                "Enable gradient checkpointing if fine-tuning",
                "Reduce maximum sequence length",
                "Use CPU offloading for large models",
                "Consider model pruning or distillation"
            ],
            fair: [
                "Test with 8-bit quantization",
                "Optimize batch size for memory efficiency",
                "Monitor for memory leaks"
            ]
        },
        relatedMetrics: ["memory_mb_avg", "cpu_percent_peak"],
        statisticalNote: "Lower is better; critical for edge deployment"
    },

    "memory_mb_avg": {
        name: "Memory Average Usage",
        category: "resources",
        unit: "MB",
        description: "Average memory consumption during inference",
        detailedExplanation: "Mean memory usage across the inference run. Indicates baseline memory requirements.",
        calculation: "mean(memory_mb_per_run)",
        formula: "MEM_avg = Σ(memory_usage_t) / N",
        interpretation: {
            excellent: { max: 1800, label: "Excellent", color: "success", description: "Very memory-efficient" },
            good: { min: 1800, max: 3600, label: "Good", color: "info", description: "Good memory efficiency" },
            fair: { min: 3600, max: 7200, label: "Fair", color: "warning", description: "Moderate memory usage" },
            poor: { min: 7200, label: "Poor", color: "error", description: "High sustained memory usage" }
        },
        recommendations: {
            poor: [
                "Use quantization (Q4_K_M, INT8, or INT4)",
                "Reduce context window size",
                "Enable memory-efficient attention mechanisms",
                "Consider smaller model variants"
            ],
            fair: [
                "Test quantized model variants",
                "Optimize memory allocation patterns",
                "Profile for memory optimization opportunities"
            ]
        },
        relatedMetrics: ["memory_mb_peak", "cpu_percent_avg"],
        statisticalNote: "Lower is better for sustained efficiency"
    },

    "output_length": {
        name: "Output Token Length",
        category: "resources",
        unit: "tokens",
        description: "Number of tokens generated in model output",
        detailedExplanation: "Length of generated response in tokens. Affects latency, memory, and cost. Highly variable based on prompt and model.",
        calculation: "len(generated_tokens)",
        formula: "output_length = count(tokens_generated)",
        interpretation: {
            excellent: { max: 100, label: "Excellent", color: "success", description: "Concise output" },
            good: { min: 100, max: 500, label: "Good", color: "info", description: "Normal output length" },
            fair: { min: 500, max: 2000, label: "Fair", color: "warning", description: "Long output" },
            poor: { min: 2000, label: "Poor", color: "error", description: "Very long output" }
        },
        recommendations: {
            poor: [
                "Set max_tokens limit to prevent runaway generation",
                "Improve prompt specificity",
                "Use stop sequences to control output length",
                "Check for repetition loops"
            ],
            fair: [
                "Review if output length matches use case needs",
                "Consider tuning temperature/top_p for brevity",
                "Monitor for verbose responses"
            ]
        },
        relatedMetrics: ["latency_p50_ms", "memory_mb_avg"],
        statisticalNote: "Longer outputs increase latency and memory usage"
    },

    // ==================== ADVANCED SPEED METRICS (Timing Breakdown) ====================
    "ttft_ms": {
        name: "Time to First Token (TTFT)",
        category: "speed",
        unit: "milliseconds",
        description: "Time elapsed from prompt submission until first token generation (prefill phase)",
        detailedExplanation: "TTFT measures the prefill/encoding phase where the model processes input and begins generation. Lower TTFT = faster response start, critical for interactive applications.",
        calculation: "timestamp_first_token - timestamp_start",
        formula: "TTFT = t_first_token - t_start",
        interpretation: {
            excellent: { max: 500, label: "Excellent", color: "success", description: "Very fast prefill, ideal for streaming" },
            good: { min: 500, max: 1500, label: "Good", color: "info", description: "Good responsiveness" },
            fair: { min: 1500, max: 3000, label: "Fair", color: "warning", description: "Noticeable delay before output starts" },
            poor: { min: 3000, label: "Poor", color: "error", description: "Slow prefill, impacts user experience" }
        },
        recommendations: {
            poor: ["Use quantized models to speed up encoding", "Enable KV cache optimization", "Reduce input context length"],
            fair: ["Test with smaller quantization levels", "Optimize prompt preprocessing"]
        },
        relatedMetrics: ["prefill_latency_ms", "decode_latency_ms"],
        statisticalNote: "TTFT dominates latency for short outputs, less important for long generations"
    },

    "prefill_latency_ms": {
        name: "Prefill Latency",
        category: "speed",
        unit: "milliseconds",
        description: "Time spent in prefill/encoding phase (same as TTFT)",
        detailedExplanation: "Prefill phase processes input prompt and prepares model state. Equivalent to TTFT. Lower values improve perceived responsiveness.",
        calculation: "timestamp_first_token - timestamp_start",
        formula: "Prefill = TTFT",
        interpretation: {
            excellent: { max: 500, label: "Excellent", color: "success", description: "Fast encoding" },
            good: { min: 500, max: 1500, label: "Good", color: "info", description: "Acceptable encoding time" },
            fair: { min: 1500, max: 3000, label: "Fair", color: "warning", description: "Slow encoding" },
            poor: { min: 3000, label: "Poor", color: "error", description: "Very slow encoding" }
        },
        recommendations: {
            poor: ["Reduce input context", "Use quantized models", "Enable prompt caching"],
            fair: ["Profile tokenization overhead", "Test with model quantization"]
        },
        relatedMetrics: ["ttft_ms", "inter_token_latency_ms"],
        statisticalNote: "Scales with input length - longer prompts = higher prefill latency"
    },

    "inter_token_latency_ms": {
        name: "Inter-Token Latency (ITL)",
        category: "speed",
        unit: "milliseconds",
        description: "Average time between consecutive token generations during decode phase",
        detailedExplanation: "ITL measures decode smoothness. Lower ITL = faster token streaming. Critical for streaming applications where consistent generation speed matters.",
        calculation: "mean(token_times[i] - token_times[i-1])",
        formula: "ITL = Σ(t_i - t_{i-1}) / (N-1)",
        interpretation: {
            excellent: { max: 50, label: "Excellent", color: "success", description: "Smooth streaming, fast tokens/sec" },
            good: { min: 50, max: 150, label: "Good", color: "info", description: "Good streaming performance" },
            fair: { min: 150, max: 300, label: "Fair", color: "warning", description: "Choppy streaming" },
            poor: { min: 300, label: "Poor", color: "error", description: "Very slow decode, poor streaming" }
        },
        recommendations: {
            poor: ["Use quantized models (Q4/Q8)", "Enable GPU acceleration", "Optimize memory bandwidth"],
            fair: ["Test with quantization", "Profile decode bottlenecks"]
        },
        relatedMetrics: ["decode_latency_ms", "tokens_per_sec"],
        statisticalNote: "Relatively constant per model - independent of output length"
    },

    "decode_latency_ms": {
        name: "Decode Latency",
        category: "speed",
        unit: "milliseconds",
        description: "Total time spent generating all output tokens (decode phase)",
        detailedExplanation: "Decode phase generates output tokens one by one. Decode latency = inter_token_latency × num_tokens. Dominates total latency for long outputs.",
        calculation: "timestamp_last_token - timestamp_first_token",
        formula: "Decode = t_last - t_first = ITL × N_tokens",
        interpretation: {
            excellent: { max: 2000, label: "Excellent", color: "success", description: "Fast token generation" },
            good: { min: 2000, max: 5000, label: "Good", color: "info", description: "Acceptable decode speed" },
            fair: { min: 5000, max: 10000, label: "Fair", color: "warning", description: "Slow decode phase" },
            poor: { min: 10000, label: "Poor", color: "error", description: "Very slow generation" }
        },
        recommendations: {
            poor: ["Use quantization", "Enable GPU if available", "Reduce max_tokens"],
            fair: ["Test with smaller model variants", "Optimize decode operations"]
        },
        relatedMetrics: ["inter_token_latency_ms", "tokens_per_sec"],
        statisticalNote: "Scales linearly with output length - longer outputs = higher decode latency"
    },

    // ==================== THERMAL & POWER METRICS ====================
    "cpu_temp_celsius_peak": {
        name: "CPU Temperature Peak",
        category: "resources",
        unit: "°C",
        description: "Peak CPU temperature observed during inference",
        detailedExplanation: "Maximum CPU temperature reached. High temperatures may cause throttling. Critical for sustained workloads and device longevity.",
        calculation: "max(cpu_temperature_during_inference)",
        formula: "T_peak = max(T_cpu(t))",
        interpretation: {
            excellent: { max: 60, label: "Excellent", color: "success", description: "Cool operation, no throttling risk" },
            good: { min: 60, max: 75, label: "Good", color: "info", description: "Warm but safe" },
            fair: { min: 75, max: 85, label: "Fair", color: "warning", description: "Hot - potential throttling risk" },
            poor: { min: 85, label: "Poor", color: "error", description: "Very hot - likely throttling" }
        },
        recommendations: {
            poor: ["Improve cooling/ventilation", "Reduce CPU load via quantization", "Enable thermal throttling limits"],
            fair: ["Monitor for sustained high temps", "Improve airflow", "Reduce batch size"]
        },
        relatedMetrics: ["cpu_temp_celsius_delta", "throttling_occurred"],
        statisticalNote: "Varies by hardware and ambient temperature"
    },

    "cpu_temp_celsius_delta": {
        name: "CPU Temperature Delta",
        category: "resources",
        unit: "°C",
        description: "Temperature increase from start to end of inference",
        detailedExplanation: "Measures heat generated by inference workload. Positive delta indicates thermal load. Large deltas suggest cooling may be inadequate.",
        calculation: "cpu_temp_end - cpu_temp_start",
        formula: "ΔT = T_end - T_start",
        interpretation: {
            excellent: { max: 5, label: "Excellent", color: "success", description: "Minimal heat generation" },
            good: { min: 5, max: 15, label: "Good", color: "info", description: "Moderate heat increase" },
            fair: { min: 15, max: 25, label: "Fair", color: "warning", description: "Significant heating" },
            poor: { min: 25, label: "Poor", color: "error", description: "Excessive heat generation" }
        },
        recommendations: {
            poor: ["Check cooling system", "Reduce workload intensity", "Improve thermal management"],
            fair: ["Monitor sustained workloads", "Consider better cooling"]
        },
        relatedMetrics: ["cpu_temp_celsius_peak", "throttling_occurred"],
        statisticalNote: "Short inferences may show minimal delta even with high CPU usage"
    },

    "gpu_temp_celsius_peak": {
        name: "GPU Temperature Peak",
        category: "resources",
        unit: "°C",
        description: "Peak GPU temperature during inference",
        detailedExplanation: "Maximum GPU temperature observed. Critical for GPU-intensive workloads. High temps can trigger throttling and reduce performance.",
        calculation: "max(gpu_temperature_during_inference)",
        formula: "T_gpu_peak = max(T_gpu(t))",
        interpretation: {
            excellent: { max: 70, label: "Excellent", color: "success", description: "Cool GPU operation" },
            good: { min: 70, max: 80, label: "Good", color: "info", description: "Normal operating temperature" },
            fair: { min: 80, max: 90, label: "Fair", color: "warning", description: "Hot - monitor for throttling" },
            poor: { min: 90, label: "Poor", color: "error", description: "Very hot - likely throttling" }
        },
        recommendations: {
            poor: ["Improve GPU cooling", "Reduce batch size", "Lower GPU clock speeds", "Check thermal paste"],
            fair: ["Monitor GPU fans", "Improve case airflow", "Reduce sustained GPU load"]
        },
        relatedMetrics: ["gpu_temp_celsius_delta", "gpu_percent_avg"],
        statisticalNote: "GPU thermal limits vary by model - check manufacturer specs"
    },

    "throttling_occurred": {
        name: "Thermal Throttling",
        category: "resources",
        unit: "bool",
        description: "Whether thermal throttling was detected during inference",
        detailedExplanation: "Indicates if CPU/GPU reduced clock speeds due to overheating. Throttling degrades performance. 1 = throttling detected, 0 = no throttling.",
        calculation: "any(thermal_throttling_events)",
        formula: "Throttling = 1 if throttled else 0",
        interpretation: {
            excellent: { max: 0.1, label: "Excellent", color: "success", description: "No throttling" },
            poor: { min: 0.1, label: "Poor", color: "error", description: "Throttling detected - performance impacted" }
        },
        recommendations: {
            poor: ["Improve cooling immediately", "Reduce workload intensity", "Check thermal limits", "Clean dust from coolers"],
        },
        relatedMetrics: ["cpu_temp_celsius_peak", "gpu_temp_celsius_peak"],
        statisticalNote: "Any throttling indicates thermal management issues"
    },

    "power_watts_avg": {
        name: "Average Power Consumption",
        category: "resources",
        unit: "W",
        description: "Average power draw during inference (if available)",
        detailedExplanation: "Measures energy efficiency. Lower power = more efficient. Important for battery-powered devices and data center costs. May not be available on all platforms.",
        calculation: "mean(power_watts_during_inference)",
        formula: "P_avg = Σ(P(t)) / N",
        interpretation: {
            excellent: { max: 50, label: "Excellent", color: "success", description: "Very power-efficient" },
            good: { min: 50, max: 150, label: "Good", color: "info", description: "Good efficiency" },
            fair: { min: 150, max: 300, label: "Fair", color: "warning", description: "Moderate power consumption" },
            poor: { min: 300, label: "Poor", color: "error", description: "High power usage" }
        },
        recommendations: {
            poor: ["Use quantized models", "Reduce batch size", "Lower GPU clock speeds", "Enable power saving mode"],
            fair: ["Test with quantization", "Optimize workload intensity"]
        },
        relatedMetrics: ["energy_joules_total", "gpu_percent_avg"],
        statisticalNote: "Not available on all platforms (requires NVIDIA GPU, Intel RAPL, or platform support)"
    },

    "energy_joules_total": {
        name: "Total Energy Consumed",
        category: "resources",
        unit: "J",
        description: "Total energy used during inference (if available)",
        detailedExplanation: "Cumulative energy consumption. Energy = Power × Time. Lower values indicate better efficiency. Important for cost and environmental impact.",
        calculation: "integral(power_watts_during_inference) or power_avg × duration",
        formula: "E = ∫P(t)dt ≈ P_avg × Δt",
        interpretation: {
            excellent: { max: 1000, label: "Excellent", color: "success", description: "Very energy-efficient" },
            good: { min: 1000, max: 5000, label: "Good", color: "info", description: "Good energy efficiency" },
            fair: { min: 5000, max: 15000, label: "Fair", color: "warning", description: "Moderate energy usage" },
            poor: { min: 15000, label: "Poor", color: "error", description: "High energy consumption" }
        },
        recommendations: {
            poor: ["Use quantization", "Reduce inference time", "Optimize for efficiency", "Consider edge-optimized models"],
            fair: ["Profile power-intensive operations", "Test with smaller models"]
        },
        relatedMetrics: ["power_watts_avg", "latency_ms"],
        statisticalNote: "Energy = Power × Time; reducing either improves efficiency"
    },

    // ==================== ADVANCED QUALITY METRICS ====================
    "bleu_score": {
        name: "BLEU Score",
        category: "quality",
        unit: "score",
        description: "Bilingual Evaluation Understudy - measures n-gram precision against reference",
        detailedExplanation: "BLEU measures how many n-grams in generated text match the reference (most common output). Higher = more consistent. Range 0-1. Industry standard for translation and generation quality.",
        calculation: "geometric_mean(precision_1gram, precision_2gram, precision_3gram, precision_4gram)",
        formula: "BLEU = BP × exp(Σw_n log(p_n))",
        interpretation: {
            excellent: { min: 0.7, label: "Excellent", color: "success", description: "High precision - very consistent outputs" },
            good: { min: 0.5, max: 0.7, label: "Good", color: "info", description: "Good consistency" },
            fair: { min: 0.3, max: 0.5, label: "Fair", color: "warning", description: "Moderate consistency" },
            poor: { max: 0.3, label: "Poor", color: "error", description: "Low consistency - outputs vary significantly" }
        },
        recommendations: {
            poor: ["Reduce temperature", "Use fixed seed", "Test with greedy decoding"],
            fair: ["Adjust sampling parameters", "Review prompt clarity"]
        },
        relatedMetrics: ["rouge_1_f1", "rouge_2_f1", "consistency_exact_match_percent"],
        statisticalNote: "BLEU emphasizes precision; may penalize valid paraphrases"
    },

    "rouge_1_f1": {
        name: "ROUGE-1 F1",
        category: "quality",
        unit: "score",
        description: "ROUGE-1 F1 score - unigram overlap between generated and reference text",
        detailedExplanation: "Measures word-level overlap. ROUGE emphasizes recall (coverage). F1 balances precision and recall. Higher = better consistency. Range 0-1.",
        calculation: "2 × (precision_1gram × recall_1gram) / (precision_1gram + recall_1gram)",
        formula: "F1 = 2PR / (P + R)",
        interpretation: {
            excellent: { min: 0.7, label: "Excellent", color: "success", description: "High word overlap - consistent content" },
            good: { min: 0.5, max: 0.7, label: "Good", color: "info", description: "Good overlap" },
            fair: { min: 0.3, max: 0.5, label: "Fair", color: "warning", description: "Moderate overlap" },
            poor: { max: 0.3, label: "Poor", color: "error", description: "Low overlap - outputs differ significantly" }
        },
        recommendations: {
            poor: ["Reduce sampling randomness", "Use more specific prompts", "Check for hallucination"],
            fair: ["Adjust temperature and top_p", "Review output diversity needs"]
        },
        relatedMetrics: ["rouge_2_f1", "rouge_L_f1", "bleu_score"],
        statisticalNote: "ROUGE-1 only considers single words - doesn't capture phrase-level consistency"
    },

    "rouge_2_f1": {
        name: "ROUGE-2 F1",
        category: "quality",
        unit: "score",
        description: "ROUGE-2 F1 score - bigram overlap (captures phrase-level consistency)",
        detailedExplanation: "Measures 2-word phrase overlap. More stringent than ROUGE-1. Better captures semantic consistency. Range 0-1.",
        calculation: "F1 score for bigram precision and recall",
        formula: "F1 = 2PR / (P + R) for bigrams",
        interpretation: {
            excellent: { min: 0.5, label: "Excellent", color: "success", description: "High phrase-level consistency" },
            good: { min: 0.3, max: 0.5, label: "Good", color: "info", description: "Good phrase overlap" },
            fair: { min: 0.15, max: 0.3, label: "Fair", color: "warning", description: "Moderate phrase consistency" },
            poor: { max: 0.15, label: "Poor", color: "error", description: "Low phrase overlap" }
        },
        recommendations: {
            poor: ["Significantly reduce temperature", "Use deterministic decoding", "Fix random seed"],
            fair: ["Lower temperature", "Review sampling strategy"]
        },
        relatedMetrics: ["rouge_1_f1", "rouge_L_f1", "bleu_score"],
        statisticalNote: "Typically lower than ROUGE-1 due to stricter matching"
    },

    "rouge_L_f1": {
        name: "ROUGE-L F1",
        category: "quality",
        unit: "score",
        description: "ROUGE-L F1 score - longest common subsequence similarity",
        detailedExplanation: "Measures longest matching word sequence (allows gaps). Captures sentence-level structure similarity. Less sensitive to word reordering. Range 0-1.",
        calculation: "F1 score based on longest common subsequence (LCS)",
        formula: "F1 = 2 × LCS_precision × LCS_recall / (LCS_precision + LCS_recall)",
        interpretation: {
            excellent: { min: 0.6, label: "Excellent", color: "success", description: "High structural consistency" },
            good: { min: 0.4, max: 0.6, label: "Good", color: "info", description: "Good structural overlap" },
            fair: { min: 0.2, max: 0.4, label: "Fair", color: "warning", description: "Moderate structural consistency" },
            poor: { max: 0.2, label: "Poor", color: "error", description: "Low structural overlap" }
        },
        recommendations: {
            poor: ["Reduce temperature", "Use structured output formatting", "Test with fixed seed"],
            fair: ["Adjust sampling parameters", "Review prompt structure"]
        },
        relatedMetrics: ["rouge_1_f1", "rouge_2_f1", "bleu_score"],
        statisticalNote: "ROUGE-L is more robust to word reordering than ROUGE-1/2"
    },

    // ==================== SEMANTIC SIMILARITY METRICS (Embeddings-based) ====================
    "semantic_similarity_mean": {
        name: "Semantic Similarity (Mean)",
        category: "quality",
        unit: "score",
        description: "Average semantic similarity using sentence embeddings (cosine similarity)",
        detailedExplanation: "Uses transformer embeddings (all-MiniLM-L6-v2) to measure actual meaning similarity. Better than BLEU/ROUGE as it captures semantic equivalence even with different words. Range 0-1, where 1.0 = identical meaning.",
        calculation: "mean(cosine_similarity(embedding_i, embedding_j) for all pairs)",
        formula: "Semantic_mean = Σ cos(emb_i, emb_j) / C(N,2)",
        interpretation: {
            excellent: { min: 0.85, label: "Excellent", color: "success", description: "Very high semantic consistency - outputs mean the same thing" },
            good: { min: 0.70, max: 0.85, label: "Good", color: "info", description: "Good semantic consistency - similar meanings" },
            fair: { min: 0.50, max: 0.70, label: "Fair", color: "warning", description: "Moderate semantic similarity - some meaning variation" },
            poor: { max: 0.50, label: "Poor", color: "error", description: "Low semantic consistency - meanings differ significantly" }
        },
        recommendations: {
            poor: [
                "Significantly reduce temperature (0.1-0.3) for consistent semantics",
                "Use fixed seed for reproducibility",
                "Review prompt clarity and specificity",
                "Check for hallucination or topic drift"
            ],
            fair: [
                "Lower temperature to reduce semantic variation",
                "Use more specific prompts to constrain output space",
                "Test with greedy decoding"
            ]
        },
        relatedMetrics: ["semantic_similarity_std", "semantic_coherence_score", "consistency_average_similarity"],
        statisticalNote: "Embedding-based metrics capture semantic meaning better than token-based metrics (BLEU/ROUGE)",
        limitations: "Requires sentence-transformers library; first run downloads 90MB model"
    },

    "semantic_similarity_min": {
        name: "Semantic Similarity (Min)",
        category: "quality",
        unit: "score",
        description: "Minimum semantic similarity - worst-case inconsistency indicator",
        detailedExplanation: "Lowest pairwise similarity score. Indicates the most divergent output pair. Low values mean at least one output has very different meaning from others.",
        calculation: "min(cosine_similarity(embedding_i, embedding_j) for all pairs)",
        formula: "Semantic_min = min{cos(emb_i, emb_j)}",
        interpretation: {
            excellent: { min: 0.75, label: "Excellent", color: "success", description: "Even worst case is semantically consistent" },
            good: { min: 0.60, max: 0.75, label: "Good", color: "info", description: "Acceptable worst-case consistency" },
            fair: { min: 0.40, max: 0.60, label: "Fair", color: "warning", description: "Some outputs diverge significantly" },
            poor: { max: 0.40, label: "Poor", color: "error", description: "At least one output has very different meaning" }
        },
        recommendations: {
            poor: [
                "Investigate which outputs are divergent",
                "Significantly reduce sampling randomness",
                "Check for hallucination or off-topic responses",
                "Use deterministic decoding"
            ],
            fair: [
                "Review outlier outputs manually",
                "Lower temperature to reduce extremes",
                "Use top_k/top_p to constrain sampling"
            ]
        },
        relatedMetrics: ["semantic_similarity_max", "semantic_outlier_count"],
        statisticalNote: "Min similarity identifies quality floor - useful for reliability assessment"
    },

    "semantic_similarity_max": {
        name: "Semantic Similarity (Max)",
        category: "quality",
        unit: "score",
        description: "Maximum semantic similarity - best-case consistency indicator",
        detailedExplanation: "Highest pairwise similarity score. Indicates the most consistent output pair. Shows best achievable consistency for this model/prompt combination.",
        calculation: "max(cosine_similarity(embedding_i, embedding_j) for all pairs)",
        formula: "Semantic_max = max{cos(emb_i, emb_j)}",
        interpretation: {
            excellent: { min: 0.95, label: "Excellent", color: "success", description: "Best pairs are nearly identical in meaning" },
            good: { min: 0.85, max: 0.95, label: "Good", color: "info", description: "Best pairs are highly consistent" },
            fair: { min: 0.70, max: 0.85, label: "Fair", color: "warning", description: "Even best pairs show variation" },
            poor: { max: 0.70, label: "Poor", color: "error", description: "No output pairs are semantically consistent" }
        },
        recommendations: {
            poor: [
                "Model may not understand prompt correctly",
                "Review prompt clarity and instructions",
                "Check for fundamental instability issues",
                "Consider different model or prompt formulation"
            ],
            fair: [
                "Even best cases show variation - review sampling parameters",
                "Check if variation is intentional (creative tasks) or problematic"
            ]
        },
        relatedMetrics: ["semantic_similarity_min", "semantic_similarity_mean"],
        statisticalNote: "Max similarity identifies quality ceiling - useful for best-case analysis"
    },

    "semantic_similarity_std": {
        name: "Semantic Similarity (Std Dev)",
        category: "quality",
        unit: "score",
        description: "Standard deviation of semantic similarities - consistency variance",
        detailedExplanation: "Measures how much pairwise similarities vary. Low std = consistent similarity across all pairs. High std = some pairs very similar, others very different (high variability).",
        calculation: "std_dev(cosine_similarities_all_pairs)",
        formula: "σ_semantic = sqrt(Σ(sim_i - sim_mean)² / N)",
        interpretation: {
            excellent: { max: 0.05, label: "Excellent", color: "success", description: "Very low variance - uniformly consistent" },
            good: { min: 0.05, max: 0.10, label: "Good", color: "info", description: "Low variance - mostly consistent" },
            fair: { min: 0.10, max: 0.20, label: "Fair", color: "warning", description: "Moderate variance - inconsistent consistency" },
            poor: { min: 0.20, label: "Poor", color: "error", description: "High variance - unpredictable consistency" }
        },
        recommendations: {
            poor: [
                "High variance suggests model instability",
                "Reduce temperature significantly",
                "Use fixed seed for reproducibility",
                "Review sampling strategy"
            ],
            fair: [
                "Moderate variance - some runs very consistent, others not",
                "Test with different random seeds",
                "Consider if variance is acceptable for use case"
            ]
        },
        relatedMetrics: ["semantic_similarity_mean", "semantic_coherence_score"],
        statisticalNote: "Low std is critical for production systems requiring predictable behavior"
    },

    "semantic_coherence_score": {
        name: "Semantic Coherence Score",
        category: "quality",
        unit: "score",
        description: "Overall semantic coherence (combines mean similarity and consistency)",
        detailedExplanation: "Composite metric: mean_similarity × (1 - std/2). Rewards both high average similarity AND low variance. Perfect score (1.0) = all outputs identical in meaning. Better overall quality indicator than mean alone.",
        calculation: "coherence = mean_similarity × (1 - std_similarity / 2)",
        formula: "Coherence = μ_sim × (1 - σ_sim/2)",
        interpretation: {
            excellent: { min: 0.80, label: "Excellent", color: "success", description: "Excellent coherence - high similarity with low variance" },
            good: { min: 0.65, max: 0.80, label: "Good", color: "info", description: "Good coherence - reliable semantic consistency" },
            fair: { min: 0.45, max: 0.65, label: "Fair", color: "warning", description: "Fair coherence - acceptable for some use cases" },
            poor: { max: 0.45, label: "Poor", color: "error", description: "Poor coherence - unreliable semantic output" }
        },
        recommendations: {
            poor: [
                "Low coherence indicates fundamental consistency issues",
                "Reduce temperature to 0.1-0.3",
                "Use deterministic decoding (greedy or beam search)",
                "Review prompt engineering",
                "Consider if model is appropriate for task"
            ],
            fair: [
                "Moderate coherence - may be acceptable for creative tasks",
                "For production use, aim for higher coherence",
                "Adjust sampling parameters to reduce variance"
            ]
        },
        relatedMetrics: ["semantic_similarity_mean", "semantic_similarity_std"],
        statisticalNote: "Coherence score is the single best metric for overall semantic quality assessment"
    },

    "semantic_outlier_count": {
        name: "Semantic Outlier Count",
        category: "quality",
        unit: "count",
        description: "Number of outputs with significantly different meaning (outliers)",
        detailedExplanation: "Counts outputs whose embeddings are >2 standard deviations from the mean embedding. Outliers have very different meaning from typical outputs. Indicates how many 'bad' or divergent outputs occurred.",
        calculation: "count(outputs where distance_from_mean > mean_distance + 2×std)",
        formula: "Outliers = |{i : ||emb_i - μ_emb|| > μ_dist + 2σ_dist}|",
        interpretation: {
            excellent: { max: 0, label: "Excellent", color: "success", description: "No semantic outliers - all outputs consistent" },
            good: { min: 0, max: 1, label: "Good", color: "info", description: "At most 1 outlier - acceptable" },
            fair: { min: 1, max: 3, label: "Fair", color: "warning", description: "Multiple outliers - review divergent outputs" },
            poor: { min: 3, label: "Poor", color: "error", description: "Many outliers - high semantic instability" }
        },
        recommendations: {
            poor: [
                "Many outliers indicate serious consistency problems",
                "Manually review outlier outputs to identify patterns",
                "Check for hallucination or topic drift",
                "Significantly reduce temperature",
                "Use nucleus sampling (top_p) to constrain output space"
            ],
            fair: [
                "Some outliers expected with creative sampling",
                "Review if outliers are problematic for your use case",
                "Consider filtering or regeneration strategies"
            ]
        },
        relatedMetrics: ["semantic_similarity_min", "semantic_similarity_std"],
        statisticalNote: "Outlier detection uses statistical distance from mean embedding - 2σ threshold is standard"
    },

    // ==================== STRESS & DEGRADATION ANALYSIS METRICS ====================
    "latency_degradation_slope_ms_per_run": {
        name: "Latency Degradation Slope",
        category: "stress",
        unit: "ms/run",
        description: "Rate of latency increase over time (linear regression slope)",
        detailedExplanation: "Measures performance degradation trend. Positive = degrading, negative = improving, ~0 = stable. Uses least squares regression over all stress test runs.",
        calculation: "linear_regression_slope(latency_values_over_time)",
        formula: "slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)",
        interpretation: {
            excellent: { min: -1, max: 1, label: "Excellent", color: "success", description: "Stable performance - no degradation" },
            good: { min: 1, max: 5, label: "Good", color: "info", description: "Minimal degradation" },
            fair: { min: 5, max: 15, label: "Fair", color: "warning", description: "Noticeable degradation over time" },
            poor: { min: 15, label: "Poor", color: "error", description: "Significant performance decline" }
        },
        recommendations: {
            poor: ["Check for memory leaks", "Profile resource usage", "Investigate garbage collection issues", "Review model caching"],
            fair: ["Monitor memory growth", "Test with longer duration"]
        },
        relatedMetrics: ["memory_growth_slope_mb_per_run", "latency_degradation_ms"],
        statisticalNote: "Slope units are ms per run - multiply by total runs to get cumulative degradation"
    },

    "memory_growth_slope_mb_per_run": {
        name: "Memory Growth Slope",
        category: "stress",
        unit: "MB/run",
        description: "Rate of memory increase over time (potential memory leak indicator)",
        detailedExplanation: "Measures memory leak severity. Positive slope = memory leak. Uses linear regression. Persistent growth indicates leak that will eventually cause OOM.",
        calculation: "linear_regression_slope(memory_values_over_time)",
        formula: "slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)",
        interpretation: {
            excellent: { min: -1, max: 1, label: "Excellent", color: "success", description: "No memory leak - stable" },
            good: { min: 1, max: 10, label: "Good", color: "info", description: "Minimal growth - acceptable" },
            fair: { min: 10, max: 50, label: "Fair", color: "warning", description: "Moderate memory leak - monitor" },
            poor: { min: 50, label: "Poor", color: "error", description: "Severe memory leak - will cause OOM" }
        },
        recommendations: {
            poor: ["Investigate memory leaks urgently", "Profile memory allocations", "Enable garbage collection logging", "Check for circular references"],
            fair: ["Monitor extended runs", "Review memory cleanup logic", "Profile heap growth"]
        },
        relatedMetrics: ["latency_degradation_slope_ms_per_run", "memory_growth_mb"],
        statisticalNote: "Multiply slope by expected run count to estimate total memory growth"
    },

    "cpu_degradation_slope_percent_per_run": {
        name: "CPU Degradation Slope",
        category: "stress",
        unit: "%/run",
        description: "Rate of CPU usage increase over time",
        detailedExplanation: "Tracks CPU efficiency degradation. Positive = increasing CPU load over time. May indicate background processes or inefficient resource cleanup.",
        calculation: "linear_regression_slope(cpu_percent_over_time)",
        formula: "slope = regression_slope(cpu_values)",
        interpretation: {
            excellent: { min: -0.1, max: 0.1, label: "Excellent", color: "success", description: "Stable CPU usage" },
            good: { min: 0.1, max: 0.5, label: "Good", color: "info", description: "Minor CPU increase" },
            fair: { min: 0.5, max: 1.5, label: "Fair", color: "warning", description: "Noticeable CPU growth" },
            poor: { min: 1.5, label: "Poor", color: "error", description: "Significant CPU degradation" }
        },
        recommendations: {
            poor: ["Profile CPU-intensive operations", "Check for background processes", "Investigate CPU leak patterns"],
            fair: ["Monitor CPU trends", "Review resource cleanup"]
        },
        relatedMetrics: ["cpu_percent_avg", "latency_degradation_slope_ms_per_run"],
        statisticalNote: "Small slopes can indicate significant issues over long durations"
    },

    "gpu_degradation_slope_percent_per_run": {
        name: "GPU Degradation Slope",
        category: "stress",
        unit: "%/run",
        description: "Rate of GPU usage increase over time",
        detailedExplanation: "Tracks GPU efficiency degradation. Positive = increasing GPU load. May indicate GPU memory fragmentation or inefficient kernel cleanup.",
        calculation: "linear_regression_slope(gpu_percent_over_time)",
        formula: "slope = regression_slope(gpu_values)",
        interpretation: {
            excellent: { min: -0.1, max: 0.1, label: "Excellent", color: "success", description: "Stable GPU usage" },
            good: { min: 0.1, max: 0.5, label: "Good", color: "info", description: "Minor GPU increase" },
            fair: { min: 0.5, max: 1.5, label: "Fair", color: "warning", description: "Noticeable GPU growth" },
            poor: { min: 1.5, label: "Poor", color: "error", description: "Significant GPU degradation" }
        },
        recommendations: {
            poor: ["Check GPU memory fragmentation", "Profile GPU kernel efficiency", "Investigate VRAM leaks"],
            fair: ["Monitor GPU trends", "Review GPU resource cleanup"]
        },
        relatedMetrics: ["gpu_percent_avg", "vram_mb_avg"],
        statisticalNote: "GPU degradation often correlates with VRAM growth"
    },

    "error_pattern_score": {
        name: "Error Clustering Score",
        category: "stress",
        unit: "score",
        description: "Degree of error clustering (0=distributed, 1=clustered)",
        detailedExplanation: "Analyzes error timing patterns. High score = errors are consecutive/clustered (suggests specific trigger). Low score = errors are randomly distributed (suggests instability). Helps diagnose root cause.",
        calculation: "max_consecutive_errors / total_errors",
        formula: "clustering = max(consecutive_errors) / total_errors",
        interpretation: {
            clustered: { min: 0.5, label: "Clustered", color: "warning", description: "Errors occur in bursts - specific trigger likely" },
            distributed: { max: 0.5, label: "Distributed", color: "info", description: "Errors are random - general instability" }
        },
        recommendations: {
            clustered: ["Investigate what triggers error clusters", "Check for resource exhaustion events", "Review error timing correlation"],
            distributed: ["Check overall system stability", "Review error randomness root cause", "Increase monitoring"]
        },
        relatedMetrics: ["error_rate_percent", "stability_passed"],
        statisticalNote: "Includes metadata: errors_in_first_third, errors_in_middle_third, errors_in_last_third for temporal analysis"
    }
};

// Category metadata for grouping and organization
const CATEGORY_METADATA = {
    speed: {
        name: "Speed & Throughput",
        icon: "⚡",
        description: "Measures how fast the model generates responses",
        color: "#00cdaf"
    },
    quality: {
        name: "Quality & Consistency",
        icon: "🎯",
        description: "Assesses output quality, consistency, and reliability",
        color: "#3aa6dd"
    },
    stress: {
        name: "Stress & Endurance",
        icon: "💪",
        description: "Tests performance under sustained load and detects degradation",
        color: "#ff8c00"
    },
    resources: {
        name: "Resource Efficiency",
        icon: "📊",
        description: "Analyzes CPU, GPU, and memory consumption",
        color: "#a870ef"
    }
};

// Helper function to get interpretation for a metric value
function getMetricInterpretation(metricName, value) {
    const metadata = METRIC_METADATA[metricName];
    if (!metadata || !metadata.interpretation) {
        return { label: "Unknown", color: "info", description: "No interpretation available" };
    }

    for (const [level, criteria] of Object.entries(metadata.interpretation)) {
        const hasMin = criteria.min !== undefined;
        const hasMax = criteria.max !== undefined;

        if (hasMin && hasMax) {
            if (value >= criteria.min && value <= criteria.max) {
                return criteria;
            }
        } else if (hasMin && value >= criteria.min) {
            return criteria;
        } else if (hasMax && value <= criteria.max) {
            return criteria;
        }
    }

    return { label: "Unknown", color: "info", description: "Value outside expected range" };
}

// Helper function to get recommendations for a metric
function getMetricRecommendations(metricName, value) {
    const interpretation = getMetricInterpretation(metricName, value);
    const metadata = METRIC_METADATA[metricName];

    if (!metadata || !metadata.recommendations) {
        return [];
    }

    const level = interpretation.label.toLowerCase();
    return metadata.recommendations[level] || [];
}

// Helper function to format metric value with unit
function formatMetricValue(value, unit) {
    if (typeof value !== 'number') return value;

    switch (unit) {
        case 'milliseconds':
        case 'ms':
            return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${value.toFixed(0)}ms`;
        case 'percentage':
        case '%':
            return `${value.toFixed(1)}%`;
        case 'MB':
            return value >= 1024 ? `${(value / 1024).toFixed(2)}GB` : `${value.toFixed(0)}MB`;
        case 'tokens':
        case 'count':
            return `${Math.round(value)}`;
        default:
            return value.toFixed(2);
    }
}

// Export for use in dashboard
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { METRIC_METADATA, CATEGORY_METADATA, getMetricInterpretation, getMetricRecommendations, formatMetricValue };
}

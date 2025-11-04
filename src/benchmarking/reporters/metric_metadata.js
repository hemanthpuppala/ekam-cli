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

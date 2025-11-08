"""
Quality & Consistency benchmark suite.

Measures output quality and consistency across multiple runs:
- Consistency (exact match percentage)
- Output variation
- Similarity scoring (character-level)
- BLEU score (n-gram precision)
- ROUGE scores (recall-oriented n-gram overlap)
- Semantic similarity (embedding-based, computed after model unload):
  * Mean/Min/Max semantic similarity
  * Standard deviation and coherence score
  * Outlier detection
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import Counter
import difflib

from src.benchmarking.suites.base_suite import BaseSuite
from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.metric_types import SuiteType, ResultStatus, MetricUnit, ModelType
from src.benchmarking.models.suite_configs import QualityConfig, quality_config_from_params
from src.benchmarking.handlers.endpoint_executor import EndpointExecutor
from src.benchmarking.datasets.defaults import get_prompts_for_suite
from src.benchmarking.utils import MemoryOptimizer
from src.benchmarking.suites._quality_helpers import (
    select_prompts_first_n,
    select_prompts_random_n,
    select_prompts_evenly_spaced
)
from loguru import logger
import random

# Optional imports for advanced quality metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    # Download required NLTK data silently
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logger.warning("NLTK not available - BLEU scores will be skipped")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score not available - ROUGE scores will be skipped")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available - semantic similarity will be skipped")


class QualitySuite(BaseSuite):
    """
    Quality & Consistency benchmark suite.

    Measures output consistency by running the same prompt multiple times
    and analyzing variation:
    - Exact match consistency (% identical outputs)
    - Average similarity (fuzzy matching)
    - Output variation (% unique outputs)
    """

    @property
    def suite_type(self) -> SuiteType:
        """Return suite type."""
        return SuiteType.QUALITY

    def __init__(self, endpoint_executor: EndpointExecutor = None, enable_progress: bool = True):
        """
        Initialize Quality suite.

        Args:
            endpoint_executor: Optional EndpointExecutor instance
            enable_progress: Whether to enable progress display (default: True)
        """
        super().__init__(enable_progress=enable_progress)
        self.executor = endpoint_executor or EndpointExecutor()

    def run(self, config: BenchmarkConfig) -> SuiteResult:
        """
        Execute quality and consistency benchmark.

        Args:
            config: BenchmarkConfig with execution parameters

        Returns:
            SuiteResult with consistency metrics
        """
        start_time = datetime.utcnow()

        try:
            # Setup
            self.setup(config)

            # Load suite-specific configuration
            suite_config = quality_config_from_params(config.parameters)
            logger.info(f"Starting Quality suite for {len(config.models)} model(s)")
            logger.info(
                f"Configuration: {suite_config.outputs_per_prompt} outputs/prompt, "
                f"{suite_config.num_warmup} warmup, {suite_config.timeout_seconds}s timeout"
            )
            logger.info(f"Quality metrics: {', '.join(suite_config.quality_metrics_to_compute)}")
            if suite_config.should_compute_metric("semantic_similarity"):
                logger.info(f"Semantic model: {suite_config.semantic_embedding_model}")

            # Prepare test data
            all_prompts = self._get_test_data(config)
            if not all_prompts:
                raise ValueError("No test prompts available")

            # Select prompts based on num_prompts_to_test
            test_prompts = self._select_prompts(all_prompts, suite_config)
            logger.info(
                f"Using {len(test_prompts)} test prompts (selected via {suite_config.prompt_selection_strategy} "
                f"from {len(all_prompts)} available) for consistency measurement"
            )

            # Run benchmarks for each model
            total_runs_attempted = 0
            total_runs_successful = 0

            # Storage for embedding computation (after model unload)
            embedding_data = []  # List of (model_id, endpoint, prompt_idx, prompt, image_path, outputs)

            for model_id in config.models:
                logger.info(f"Benchmarking model: {model_id}")

                try:
                    for endpoint in config.endpoints.get(model_id, []):
                        logger.info(f"  Endpoint: {endpoint}")

                        # For quality, we run the same prompt multiple times (outputs_per_prompt)
                        # 1 prompt = multiple runs for consistency testing
                        for prompt_idx, prompt in enumerate(test_prompts):
                            logger.info(f"  Testing prompt {prompt_idx + 1}/{len(test_prompts)}: '{prompt[:50]}...'")

                            # Get image path for VLMs - 1:1 mapping with prompts
                            if config.model_type == ModelType.VLM:
                                images = config.test_data.get("images", [])
                                if images and prompt_idx < len(images):
                                    image_path = images[prompt_idx]
                                elif images:
                                    # Fallback to cycling if more prompts than images
                                    image_path = images[prompt_idx % len(images)]
                                else:
                                    image_path = "-"
                            else:
                                image_path = "-"

                            # Collect outputs for this prompt
                            outputs = []

                            # Warmup runs (if configured)
                            if suite_config.num_warmup > 0:
                                logger.info(f"  Running {suite_config.num_warmup} warmup runs...")
                                for warmup_num in range(1, suite_config.num_warmup + 1):
                                    output = self._execute_quality_run(
                                        model_id=model_id,
                                        endpoint=endpoint,
                                        prompt=prompt,
                                        run_number=warmup_num,
                                        is_warmup=True,
                                        config=config,
                                        suite_config=suite_config
                                    )
                                    if output:
                                        outputs.append(output)

                            # Counted runs - measure consistency
                            # NOTE: outputs_per_prompt is the number of outputs to generate for EACH prompt
                            logger.info(f"  Running {suite_config.outputs_per_prompt} outputs for this prompt...")
                            counted_outputs = []
                            for run_num in range(1, suite_config.outputs_per_prompt + 1):
                                total_runs_attempted += 1

                                output = self._execute_quality_run(
                                    model_id=model_id,
                                    endpoint=endpoint,
                                    prompt=prompt,
                                    run_number=run_num,
                                    is_warmup=False,
                                    config=config,
                                    suite_config=suite_config
                                )

                                if output:
                                    counted_outputs.append(output)
                                    total_runs_successful += 1

                            # Compute consistency metrics for counted runs
                            if counted_outputs:
                                self._compute_consistency_metrics(
                                    outputs=counted_outputs,
                                    model_id=model_id,
                                    endpoint=endpoint,
                                    prompt=prompt,
                                    image_path=image_path,
                                    prompt_idx=prompt_idx,
                                    is_warmup=False,
                                    suite_config=suite_config
                                )

                                # Store outputs for embedding computation (after model unload)
                                embedding_data.append({
                                    'model_id': model_id,
                                    'endpoint': endpoint,
                                    'prompt_idx': prompt_idx,
                                    'prompt': prompt,
                                    'image_path': image_path,
                                    'outputs': counted_outputs
                                })

                finally:
                    # Unload model after all runs complete
                    if hasattr(self.executor, 'llm_handler') and \
                       hasattr(self.executor.llm_handler, 'provider_bridge') and \
                       self.executor.llm_handler.provider_bridge:
                        logger.info(f"  Unloading model: {model_id}")
                        self.executor.llm_handler.provider_bridge.unload_model(model_id)
                    elif hasattr(self.executor, 'vlm_handler') and \
                         hasattr(self.executor.vlm_handler, 'provider_bridge') and \
                         self.executor.vlm_handler.provider_bridge:
                        logger.info(f"  Unloading model: {model_id}")
                        self.executor.vlm_handler.provider_bridge.unload_model(model_id)

            # After ALL models unloaded, compute embedding-based semantic similarity (if configured)
            if EMBEDDINGS_AVAILABLE and embedding_data and suite_config.should_compute_metric("semantic_similarity"):
                logger.info(f"Computing semantic similarity for {len(embedding_data)} prompt groups...")
                self._compute_semantic_similarity_metrics(embedding_data, suite_config)

            # Determine status
            if total_runs_successful == total_runs_attempted:
                status = ResultStatus.SUCCESS
            elif total_runs_successful == 0:
                status = ResultStatus.FAILED
            else:
                status = ResultStatus.PARTIAL

            # Create result
            end_time = datetime.utcnow()
            result = self._create_result(start_time, end_time, status)

            logger.info(
                f"Quality suite complete: {total_runs_successful}/{total_runs_attempted} successful"
            )

            return result

        except Exception as e:
            logger.error(f"Quality suite failed: {str(e)}")
            self.on_run_error(e)

            end_time = datetime.utcnow()
            return self._create_result(start_time, end_time, ResultStatus.FAILED)

        finally:
            self.teardown()

    def _execute_quality_run(
        self,
        model_id: str,
        endpoint: str,
        prompt: str,
        run_number: int,
        is_warmup: bool,
        config: BenchmarkConfig,
        suite_config: QualityConfig
    ) -> str:
        """
        Execute a single quality benchmark run with memory optimization.

        Args:
            model_id: Model being tested
            endpoint: Endpoint being used
            prompt: Input prompt
            run_number: Run number
            is_warmup: Whether this is a warmup run
            config: Benchmark configuration

        Returns:
            Output text (or None if failed)
        """
        try:
            # Memory optimization: Check and cleanup
            pressure = MemoryOptimizer.memory_pressure_level()
            if pressure == "critical":
                logger.warning(f"Skipping quality run {run_number}: Critical memory")
                MemoryOptimizer.aggressive_cleanup()
                return None
            elif pressure in ["high", "critical"]:
                MemoryOptimizer.aggressive_cleanup()

            # Prepare input data and track for CSV output
            if config.model_type == ModelType.LLM:
                input_data = prompt
                input_image_path = "-"  # LLMs don't use images
            else:  # VLM
                images = config.test_data.get("images", [])
                if not images:
                    logger.warning("No images provided for VLM benchmark")
                    return None

                input_image_path = images[run_number % len(images)]
                input_data = {
                    "prompt": prompt,
                    "image_path": input_image_path
                }

            # Execute inference with latency tracking
            import time
            start_time = time.perf_counter()

            result = self.executor.execute(
                model_type=config.model_type,
                model_id=model_id,
                endpoint=endpoint,
                input_data=input_data,
                parameters=config.parameters,
                timeout=suite_config.timeout_seconds
            )

            # Calculate latency
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Post-run cleanup
            MemoryOptimizer.aggressive_cleanup()

            # Get output text (untruncated)
            output = result.get("output", "")

            # Create metadata dict with full input/output details for CSV
            run_metadata = {
                "input_prompt": prompt,
                "input_image_path": input_image_path,
                "raw_response": output
            }

            # Record inference latency (consistent with other suites)
            self._record_metric(
                name="latency_ms",
                value=latency_ms,
                unit=MetricUnit.MILLISECONDS,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup,
                metadata=run_metadata
            )

            # Record output length for reference
            output_length = len(output)
            self._record_metric(
                name="output_length",
                value=float(output_length),
                unit=MetricUnit.COUNT,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup,
                metadata=run_metadata
            )

            return output

        except Exception as e:
            logger.error(f"Quality run {run_number} failed: {str(e)}")
            self._record_error(model_id, endpoint, run_number, str(e))
            return None

    def _compute_consistency_metrics(
        self,
        outputs: List[str],
        model_id: str,
        endpoint: str,
        prompt: str,
        image_path: str,
        prompt_idx: int,
        is_warmup: bool,
        suite_config: QualityConfig
    ):
        """
        Compute consistency metrics from multiple outputs.

        Args:
            outputs: List of output strings from same prompt
            model_id: Model ID
            endpoint: Endpoint name
            prompt: Input prompt text
            image_path: Input image path (or "-" for LLMs)
            prompt_idx: Prompt index
            is_warmup: Whether these are warmup runs
        """
        if not outputs:
            logger.warning("No outputs to compute consistency metrics")
            return

        # Create base metadata with input/output details for CSV
        base_metadata = {
            "input_prompt": prompt,
            "input_image_path": image_path,
            "raw_response": "; ".join(outputs),  # Join all outputs for context
            "total_outputs": len(outputs),
            "unique_outputs": len(set(outputs))
        }

        # 1. Exact match consistency (% identical outputs)
        output_counts = Counter(outputs)
        most_common_output, most_common_count = output_counts.most_common(1)[0]
        exact_match_percent = (most_common_count / len(outputs)) * 100.0

        self._record_metric(
            name="consistency_exact_match_percent",
            value=exact_match_percent,
            unit=MetricUnit.PERCENT,
            run_number=prompt_idx + 1,
            model_id=model_id,
            endpoint=endpoint,
            is_warmup=is_warmup,
            metadata=base_metadata
        )

        # 2. Output variation (% unique outputs)
        variation_percent = (len(output_counts) / len(outputs)) * 100.0

        variation_metadata = base_metadata.copy()
        variation_metadata["unique_count"] = len(output_counts)
        variation_metadata["total_count"] = len(outputs)

        self._record_metric(
            name="output_variation_percent",
            value=variation_percent,
            unit=MetricUnit.PERCENT,
            run_number=prompt_idx + 1,
            model_id=model_id,
            endpoint=endpoint,
            is_warmup=is_warmup,
            metadata=variation_metadata
        )

        # 3. Average similarity (fuzzy matching)
        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                similarity = difflib.SequenceMatcher(None, outputs[i], outputs[j]).ratio()
                similarities.append(similarity)

        if similarities:
            avg_similarity = (sum(similarities) / len(similarities)) * 100.0
        else:
            avg_similarity = 100.0  # Single output = perfect consistency

        similarity_metadata = base_metadata.copy()
        similarity_metadata["comparisons_made"] = len(similarities)

        self._record_metric(
            name="consistency_average_similarity",
            value=avg_similarity,
            unit=MetricUnit.PERCENT,
            run_number=prompt_idx + 1,
            model_id=model_id,
            endpoint=endpoint,
            is_warmup=is_warmup,
            metadata=similarity_metadata
        )

        # 4. BLEU Score (n-gram precision against reference) - if configured
        if suite_config.should_compute_metric("bleu") and BLEU_AVAILABLE and len(outputs) > 1:
            # Use most common output as reference
            reference = most_common_output
            bleu_scores = self._compute_bleu_scores(outputs, reference)

            if bleu_scores:
                avg_bleu = sum(bleu_scores) / len(bleu_scores)
                bleu_metadata = base_metadata.copy()
                bleu_metadata["reference_output"] = reference[:100]  # Truncate for CSV

                self._record_metric(
                    name="bleu_score",
                    value=avg_bleu,
                    unit=MetricUnit.SCORE,
                    run_number=prompt_idx + 1,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=is_warmup,
                    metadata=bleu_metadata
                )

        # 5. ROUGE Scores (recall-oriented n-gram overlap) - if configured
        if suite_config.should_compute_metric("rouge") and ROUGE_AVAILABLE and len(outputs) > 1:
            # Use most common output as reference
            reference = most_common_output
            rouge_scores = self._compute_rouge_scores(outputs, reference)

            if rouge_scores:
                rouge_metadata = base_metadata.copy()
                rouge_metadata["reference_output"] = reference[:100]

                # Record ROUGE-1, ROUGE-2, ROUGE-L F1 scores
                for rouge_type, score in rouge_scores.items():
                    self._record_metric(
                        name=f"rouge_{rouge_type}_f1",
                        value=score,
                        unit=MetricUnit.SCORE,
                        run_number=prompt_idx + 1,
                        model_id=model_id,
                        endpoint=endpoint,
                        is_warmup=is_warmup,
                        metadata=rouge_metadata
                    )

        logger.info(
            f"  Consistency: {exact_match_percent:.1f}% exact, "
            f"{avg_similarity:.1f}% avg similarity, "
            f"{len(output_counts)} unique outputs"
        )

    def _get_test_data(self, config: BenchmarkConfig) -> List[str]:
        """
        Get test prompts from config or defaults.

        Args:
            config: Benchmark configuration

        Returns:
            List of test prompts
        """
        # Try to get from config
        if "prompts" in config.test_data and config.test_data["prompts"]:
            return config.test_data["prompts"]

        # Fall back to defaults
        model_type_str = "llm" if config.model_type == ModelType.LLM else "vlm"
        return get_prompts_for_suite("quality", model_type_str)

    def _select_prompts(self, prompts: List[str], suite_config: QualityConfig) -> List[str]:
        """
        Select prompts based on num_prompts_to_test and selection strategy.

        Args:
            prompts: All available prompts
            suite_config: Quality configuration

        Returns:
            Selected prompts
        """
        n = suite_config.num_prompts_to_test
        strategy = suite_config.prompt_selection_strategy

        if strategy == "first_n":
            return select_prompts_first_n(prompts, n)
        elif strategy == "random_n":
            return select_prompts_random_n(prompts, n)
        elif strategy == "evenly_spaced":
            return select_prompts_evenly_spaced(prompts, n)
        else:
            logger.warning(f"Unknown prompt selection strategy: {strategy}, using first_n")
            return select_prompts_first_n(prompts, n)

    def _execute_inference(
        self,
        model_id: str,
        endpoint: str,
        input_data: Any,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute actual inference (required by BaseSuite).

        This is delegated to the endpoint executor.

        Args:
            model_id: Model ID
            endpoint: Endpoint name
            input_data: Input data
            parameters: Inference parameters

        Returns:
            Model output
        """
        result = self.executor.execute(
            model_type=self._config.model_type if self._config else ModelType.LLM,
            model_id=model_id,
            endpoint=endpoint,
            input_data=input_data,
            parameters=parameters,
            timeout=600
        )

        return result.get("output", "")

    def _compute_bleu_scores(self, outputs: List[str], reference: str) -> List[float]:
        """
        Compute BLEU scores for outputs against a reference.

        Args:
            outputs: List of generated outputs
            reference: Reference text (typically most common output)

        Returns:
            List of BLEU scores (0-1 scale)
        """
        if not BLEU_AVAILABLE:
            return []

        scores = []
        try:
            # Tokenize reference
            reference_tokens = word_tokenize(reference.lower())

            # Use smoothing function to handle edge cases
            smoothing = SmoothingFunction().method1

            for output in outputs:
                if output == reference:
                    # Perfect match
                    scores.append(1.0)
                else:
                    try:
                        # Tokenize candidate
                        candidate_tokens = word_tokenize(output.lower())

                        # Compute BLEU score (using BLEU-4 by default)
                        # reference should be a list of reference translations
                        bleu = sentence_bleu(
                            [reference_tokens],
                            candidate_tokens,
                            smoothing_function=smoothing
                        )
                        scores.append(bleu)
                    except Exception as e:
                        logger.warning(f"BLEU computation failed for output: {e}")
                        scores.append(0.0)

            logger.debug(f"Computed {len(scores)} BLEU scores, avg: {sum(scores)/len(scores):.3f}")

        except Exception as e:
            logger.error(f"BLEU score computation failed: {e}")

        return scores

    def _compute_rouge_scores(self, outputs: List[str], reference: str) -> Dict[str, float]:
        """
        Compute ROUGE scores for outputs against a reference.

        Args:
            outputs: List of generated outputs
            reference: Reference text (typically most common output)

        Returns:
            Dictionary with average ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        if not ROUGE_AVAILABLE:
            return {}

        try:
            # Initialize ROUGE scorer for ROUGE-1, ROUGE-2, and ROUGE-L
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            # Collect scores for each output
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []

            for output in outputs:
                if output == reference:
                    # Perfect match
                    rouge1_scores.append(1.0)
                    rouge2_scores.append(1.0)
                    rougeL_scores.append(1.0)
                else:
                    try:
                        scores = scorer.score(reference, output)

                        # Extract F1 scores
                        rouge1_scores.append(scores['rouge1'].fmeasure)
                        rouge2_scores.append(scores['rouge2'].fmeasure)
                        rougeL_scores.append(scores['rougeL'].fmeasure)

                    except Exception as e:
                        logger.warning(f"ROUGE computation failed for output: {e}")
                        rouge1_scores.append(0.0)
                        rouge2_scores.append(0.0)
                        rougeL_scores.append(0.0)

            # Compute averages
            result = {
                '1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
                '2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
                'L': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
            }

            logger.debug(f"ROUGE scores - R1: {result['1']:.3f}, R2: {result['2']:.3f}, RL: {result['L']:.3f}")

            return result

        except Exception as e:
            logger.error(f"ROUGE score computation failed: {e}")
            return {}

    def _compute_semantic_similarity_metrics(self, embedding_data: List[Dict[str, Any]], suite_config: QualityConfig) -> None:
        """
        Compute comprehensive semantic similarity metrics using embeddings.

        This is called AFTER model unload to free memory first.

        Metrics computed:
        - semantic_similarity_mean: Average cosine similarity across all output pairs
        - semantic_similarity_min: Minimum similarity (worst-case inconsistency)
        - semantic_similarity_max: Maximum similarity (best-case consistency)
        - semantic_similarity_std: Standard deviation of similarities
        - semantic_coherence_score: Overall semantic coherence (0-1)
        - semantic_outlier_count: Number of outputs significantly different from mean

        Args:
            embedding_data: List of dicts with model_id, endpoint, prompt_idx, outputs
            suite_config: Quality suite configuration
        """
        if not EMBEDDINGS_AVAILABLE:
            return

        try:
            # Load configurable embedding model
            logger.info(f"Loading sentence transformer model ({suite_config.semantic_embedding_model})...")
            model = SentenceTransformer(suite_config.semantic_embedding_model)
            logger.info("Embedding model loaded successfully")

            for data in embedding_data:
                model_id = data['model_id']
                endpoint = data['endpoint']
                prompt_idx = data['prompt_idx']
                prompt = data['prompt']
                image_path = data['image_path']
                outputs = data['outputs']

                if len(outputs) < 2:
                    logger.debug(f"Skipping semantic similarity for {model_id} (less than 2 outputs)")
                    continue

                # Encode all outputs to embeddings
                logger.debug(f"Computing embeddings for {len(outputs)} outputs...")
                embeddings = model.encode(outputs, show_progress_bar=False, convert_to_numpy=True)

                # Compute pairwise cosine similarities
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        # Cosine similarity = dot product of normalized vectors
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        similarities.append(float(sim))

                if not similarities:
                    continue

                # Convert to numpy for statistics
                similarities = np.array(similarities)

                # Compute comprehensive metrics
                mean_similarity = float(np.mean(similarities))
                min_similarity = float(np.min(similarities))
                max_similarity = float(np.max(similarities))
                std_similarity = float(np.std(similarities))

                # Coherence score: weighted combination of mean and consistency
                # High coherence = high mean AND low std
                coherence_score = mean_similarity * (1 - std_similarity / 2)  # Range 0-1

                # Outlier detection: embeddings far from mean embedding
                mean_embedding = np.mean(embeddings, axis=0)
                distances_from_mean = []
                for emb in embeddings:
                    dist = np.linalg.norm(emb - mean_embedding)
                    distances_from_mean.append(dist)

                # Outliers: distances > mean + N*std (configurable threshold)
                distances = np.array(distances_from_mean)
                outlier_threshold = np.mean(distances) + suite_config.semantic_outlier_threshold_sigma * np.std(distances)
                outlier_count = int(np.sum(distances > outlier_threshold))

                # Create metadata
                base_metadata = {
                    "input_prompt": prompt,
                    "input_image_path": image_path,
                    "total_outputs": len(outputs),
                    "unique_outputs": len(set(outputs)),
                    "pairwise_comparisons": len(similarities)
                }

                # Record mean similarity
                self._record_metric(
                    name="semantic_similarity_mean",
                    value=mean_similarity,
                    unit=MetricUnit.SCORE,
                    run_number=prompt_idx + 1,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata=base_metadata
                )

                # Record min similarity (worst-case inconsistency)
                self._record_metric(
                    name="semantic_similarity_min",
                    value=min_similarity,
                    unit=MetricUnit.SCORE,
                    run_number=prompt_idx + 1,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata={**base_metadata, "interpretation": "Worst-case semantic inconsistency"}
                )

                # Record max similarity (best-case consistency)
                self._record_metric(
                    name="semantic_similarity_max",
                    value=max_similarity,
                    unit=MetricUnit.SCORE,
                    run_number=prompt_idx + 1,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata={**base_metadata, "interpretation": "Best-case semantic consistency"}
                )

                # Record standard deviation
                self._record_metric(
                    name="semantic_similarity_std",
                    value=std_similarity,
                    unit=MetricUnit.SCORE,
                    run_number=prompt_idx + 1,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata={**base_metadata, "interpretation": "Consistency variance (lower = more consistent)"}
                )

                # Record coherence score
                self._record_metric(
                    name="semantic_coherence_score",
                    value=coherence_score,
                    unit=MetricUnit.SCORE,
                    run_number=prompt_idx + 1,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata={**base_metadata, "interpretation": "Overall semantic coherence (0-1, higher = better)"}
                )

                # Record outlier count
                self._record_metric(
                    name="semantic_outlier_count",
                    value=float(outlier_count),
                    unit=MetricUnit.COUNT,
                    run_number=prompt_idx + 1,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata={
                        **base_metadata,
                        "outlier_threshold_distance": float(outlier_threshold),
                        "interpretation": "Number of semantically divergent outputs"
                    }
                )

                logger.info(
                    f"  Semantic similarity for {model_id}: mean={mean_similarity:.3f}, "
                    f"std={std_similarity:.3f}, coherence={coherence_score:.3f}, outliers={outlier_count}"
                )

            logger.info("Semantic similarity computation complete")

        except Exception as e:
            logger.error(f"Failed to compute semantic similarity: {str(e)}")
            logger.error("This is non-fatal - other quality metrics are still available")

    def __repr__(self) -> str:
        """String representation."""
        return f"QualitySuite(executor={self.executor})"

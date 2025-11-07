"""
Graph generation for benchmark results.

Creates both interactive (Plotly) and static (Matplotlib) visualizations
tailored to each benchmark suite type.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Plotly imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not installed - interactive graphs disabled")

# Matplotlib imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not installed - static graphs disabled")

from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.metric_types import SuiteType


class GraphGenerator:
    """
    Generates graphs for benchmark results.

    Features:
    - Suite-specific visualizations
    - Both interactive (Plotly HTML) and static (Matplotlib PNG)
    - Automatic layout and styling
    - Saves to benchmark results directory
    """

    def __init__(self):
        """Initialize graph generator."""
        self.plotly_available = PLOTLY_AVAILABLE
        self.matplotlib_available = MATPLOTLIB_AVAILABLE

        if not self.plotly_available and not self.matplotlib_available:
            logger.error("No graphing libraries available! Install plotly or matplotlib.")

    def generate_graphs(
        self,
        result: SuiteResult,
        output_dir: Path,
        metrics_df: Optional[pd.DataFrame] = None,
        aggregates_df: Optional[pd.DataFrame] = None
    ) -> List[Path]:
        """
        Generate all graphs for a benchmark result.

        Args:
            result: SuiteResult to visualize
            output_dir: Directory to save graphs
            metrics_df: Pre-loaded metrics DataFrame (optional)
            aggregates_df: Pre-loaded aggregates DataFrame (optional)

        Returns:
            List of paths to generated graph files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        logger.info(f"Generating graphs for {result.suite_type.value} suite...")

        # Load CSVs if not provided
        if metrics_df is None:
            metrics_path = output_dir / "result_metrics.csv"
            if metrics_path.exists():
                metrics_df = pd.read_csv(metrics_path)

        if aggregates_df is None:
            aggregates_path = output_dir / "result_aggregates.csv"
            if aggregates_path.exists():
                aggregates_df = pd.read_csv(aggregates_path)

        # Generate suite-specific graphs
        if result.suite_type == SuiteType.SPEED:
            generated_files.extend(self._generate_speed_graphs(
                result, output_dir, metrics_df, aggregates_df
            ))
        elif result.suite_type == SuiteType.RESOURCES:
            generated_files.extend(self._generate_resources_graphs(
                result, output_dir, metrics_df, aggregates_df
            ))
        elif result.suite_type == SuiteType.COMPLETE:
            generated_files.extend(self._generate_complete_graphs(
                result, output_dir, metrics_df, aggregates_df
            ))

        logger.info(f"Generated {len(generated_files)} graph files")
        return generated_files

    def _generate_speed_graphs(
        self,
        result: SuiteResult,
        output_dir: Path,
        metrics_df: pd.DataFrame,
        aggregates_df: pd.DataFrame
    ) -> List[Path]:
        """Generate graphs specific to Speed suite."""
        files = []

        if metrics_df is None or aggregates_df is None:
            logger.warning("No data available for Speed graphs")
            return files

        # Filter for per-model aggregates (counted runs only)
        per_model = aggregates_df[
            (aggregates_df['aggregation_level'] == 'per_model') &
            (aggregates_df['aggregation_type'] == 'counted')
        ]

        # 1. Latency Box Plot (per model)
        files.extend(self._create_latency_boxplot(
            metrics_df, output_dir, "speed_latency_comparison"
        ))

        # 2. Warmup vs Counted Bar Chart
        files.extend(self._create_warmup_vs_counted_bar(
            aggregates_df, output_dir, "speed_warmup_impact"
        ))

        # 3. Latency Over Runs Line Chart
        files.extend(self._create_latency_over_runs(
            metrics_df, output_dir, "speed_latency_trend"
        ))

        # 4. Output Length vs Latency Scatter
        files.extend(self._create_output_vs_latency_scatter(
            metrics_df, output_dir, "speed_output_correlation"
        ))

        # 5. Percentiles Bar Chart
        if not per_model.empty:
            files.extend(self._create_percentiles_bar(
                per_model, output_dir, "speed_percentiles"
            ))

        return files

    def _generate_resources_graphs(
        self,
        result: SuiteResult,
        output_dir: Path,
        metrics_df: pd.DataFrame,
        aggregates_df: pd.DataFrame
    ) -> List[Path]:
        """Generate graphs specific to Resources suite."""
        files = []

        if metrics_df is None or aggregates_df is None:
            logger.warning("No data available for Resources graphs")
            return files

        # Filter for per-model aggregates (counted runs only)
        per_model = aggregates_df[
            (aggregates_df['aggregation_level'] == 'per_model') &
            (aggregates_df['aggregation_type'] == 'counted')
        ]

        # 1. CPU/Memory Utilization Bars
        files.extend(self._create_resource_utilization_bars(
            per_model, output_dir, "resources_utilization"
        ))

        # 2. Resource Efficiency Scatter
        files.extend(self._create_resource_efficiency_scatter(
            per_model, output_dir, "resources_efficiency"
        ))

        # 3. Resource Usage Over Time
        files.extend(self._create_resource_over_time(
            metrics_df, output_dir, "resources_timeline"
        ))

        # 4. Memory Consumption Comparison
        files.extend(self._create_memory_comparison(
            per_model, output_dir, "resources_memory"
        ))

        return files

    def _generate_complete_graphs(
        self,
        result: SuiteResult,
        output_dir: Path,
        metrics_df: pd.DataFrame,
        aggregates_df: pd.DataFrame
    ) -> List[Path]:
        """Generate graphs for Complete suite (combines all sub-suites)."""
        files = []

        if metrics_df is None or aggregates_df is None:
            logger.warning("No data available for Complete graphs")
            return files

        # Generate Speed sub-suite graphs
        files.extend(self._generate_speed_graphs(
            result, output_dir, metrics_df, aggregates_df
        ))

        # Generate Resources sub-suite graphs
        files.extend(self._generate_resources_graphs(
            result, output_dir, metrics_df, aggregates_df
        ))

        # Generate combined dashboard
        files.extend(self._create_combined_dashboard(
            metrics_df, aggregates_df, output_dir, "complete_dashboard"
        ))

        return files

    # ========== Individual Graph Creators ==========

    def _create_latency_boxplot(
        self, metrics_df: pd.DataFrame, output_dir: Path, filename: str
    ) -> List[Path]:
        """Create box plot comparing latency across models."""
        files = []

        # Filter for latency metrics (counted runs only)
        latency_df = metrics_df[
            (metrics_df['metric_name'] == 'latency_ms') &
            (metrics_df['is_warmup'] == False)
        ]

        if latency_df.empty:
            return files

        # Plotly version (interactive)
        if self.plotly_available:
            fig = go.Figure()

            for model_id in latency_df['model_id'].unique():
                model_data = latency_df[latency_df['model_id'] == model_id]

                fig.add_trace(go.Box(
                    y=model_data['value'],
                    name=self._shorten_model_name(model_id),
                    boxmean='sd'  # Show mean and std dev
                ))

            fig.update_layout(
                title="Model Latency Comparison (Counted Runs)",
                yaxis_title="Latency (ms)",
                xaxis_title="Model",
                showlegend=False,
                height=500
            )

            html_path = output_dir / f"{filename}.html"
            fig.write_html(str(html_path))
            files.append(html_path)
            logger.info(f"Created interactive box plot: {html_path}")

        # Matplotlib version (static)
        if self.matplotlib_available:
            plt.figure(figsize=(10, 6))

            # Prepare data for box plot
            model_ids = latency_df['model_id'].unique()
            data_by_model = [
                latency_df[latency_df['model_id'] == mid]['value'].values
                for mid in model_ids
            ]

            plt.boxplot(data_by_model, labels=[
                self._shorten_model_name(mid) for mid in model_ids
            ])

            plt.ylabel('Latency (ms)')
            plt.xlabel('Model')
            plt.title('Model Latency Comparison (Counted Runs)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            png_path = output_dir / f"{filename}.png"
            plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close()
            files.append(png_path)
            logger.info(f"Created static box plot: {png_path}")

        return files

    def _create_warmup_vs_counted_bar(
        self, aggregates_df: pd.DataFrame, output_dir: Path, filename: str
    ) -> List[Path]:
        """Create grouped bar chart comparing warmup vs counted run latencies."""
        files = []

        # Filter for latency aggregates per model
        latency_agg = aggregates_df[
            (aggregates_df['aggregation_level'] == 'per_model') &
            (aggregates_df['metric_name'] == 'latency_ms')
        ]

        if latency_agg.empty:
            return files

        # Pivot data
        warmup_data = latency_agg[latency_agg['aggregation_type'] == 'warmup']
        counted_data = latency_agg[latency_agg['aggregation_type'] == 'counted']

        # Plotly version
        if self.plotly_available:
            fig = go.Figure()

            model_ids = counted_data['model_id'].unique()

            # Warmup bars
            warmup_means = [
                warmup_data[warmup_data['model_id'] == mid]['mean'].values[0]
                if len(warmup_data[warmup_data['model_id'] == mid]) > 0 else 0
                for mid in model_ids
            ]

            # Counted bars
            counted_means = [
                counted_data[counted_data['model_id'] == mid]['mean'].values[0]
                for mid in model_ids
            ]

            fig.add_trace(go.Bar(
                name='Warmup Runs',
                x=[self._shorten_model_name(mid) for mid in model_ids],
                y=warmup_means,
                marker_color='coral'
            ))

            fig.add_trace(go.Bar(
                name='Counted Runs',
                x=[self._shorten_model_name(mid) for mid in model_ids],
                y=counted_means,
                marker_color='lightseagreen'
            ))

            fig.update_layout(
                title="Warmup vs Counted Run Latency",
                yaxis_title="Mean Latency (ms)",
                xaxis_title="Model",
                barmode='group',
                height=500
            )

            html_path = output_dir / f"{filename}.html"
            fig.write_html(str(html_path))
            files.append(html_path)
            logger.info(f"Created warmup comparison: {html_path}")

        # Matplotlib version
        if self.matplotlib_available:
            model_ids = counted_data['model_id'].unique()

            warmup_means = np.array([
                warmup_data[warmup_data['model_id'] == mid]['mean'].values[0]
                if len(warmup_data[warmup_data['model_id'] == mid]) > 0 else 0
                for mid in model_ids
            ])

            counted_means = np.array([
                counted_data[counted_data['model_id'] == mid]['mean'].values[0]
                for mid in model_ids
            ])

            x = np.arange(len(model_ids))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, warmup_means, width, label='Warmup Runs', color='coral')
            ax.bar(x + width/2, counted_means, width, label='Counted Runs', color='lightseagreen')

            ax.set_ylabel('Mean Latency (ms)')
            ax.set_xlabel('Model')
            ax.set_title('Warmup vs Counted Run Latency')
            ax.set_xticks(x)
            ax.set_xticklabels([self._shorten_model_name(mid) for mid in model_ids], rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()

            png_path = output_dir / f"{filename}.png"
            plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close()
            files.append(png_path)
            logger.info(f"Created static warmup comparison: {png_path}")

        return files

    def _create_latency_over_runs(
        self, metrics_df: pd.DataFrame, output_dir: Path, filename: str
    ) -> List[Path]:
        """Create line chart showing latency trend over run numbers."""
        files = []

        latency_df = metrics_df[metrics_df['metric_name'] == 'latency_ms']

        if latency_df.empty:
            return files

        # Plotly version
        if self.plotly_available:
            fig = go.Figure()

            for model_id in latency_df['model_id'].unique():
                model_data = latency_df[latency_df['model_id'] == model_id].sort_values('run_number')

                # Separate warmup and counted
                warmup = model_data[model_data['is_warmup'] == True]
                counted = model_data[model_data['is_warmup'] == False]

                short_name = self._shorten_model_name(model_id)

                # Warmup runs (dashed line)
                if not warmup.empty:
                    fig.add_trace(go.Scatter(
                        x=warmup['run_number'],
                        y=warmup['value'],
                        name=f'{short_name} (warmup)',
                        mode='lines+markers',
                        line=dict(dash='dash'),
                        opacity=0.5
                    ))

                # Counted runs (solid line)
                if not counted.empty:
                    fig.add_trace(go.Scatter(
                        x=counted['run_number'],
                        y=counted['value'],
                        name=short_name,
                        mode='lines+markers'
                    ))

            fig.update_layout(
                title="Latency Trend Over Runs",
                yaxis_title="Latency (ms)",
                xaxis_title="Run Number",
                height=500
            )

            html_path = output_dir / f"{filename}.html"
            fig.write_html(str(html_path))
            files.append(html_path)
            logger.info(f"Created latency trend chart: {html_path}")

        # Matplotlib version
        if self.matplotlib_available:
            plt.figure(figsize=(10, 6))

            for model_id in latency_df['model_id'].unique():
                model_data = latency_df[latency_df['model_id'] == model_id].sort_values('run_number')

                warmup = model_data[model_data['is_warmup'] == True]
                counted = model_data[model_data['is_warmup'] == False]

                short_name = self._shorten_model_name(model_id)

                if not warmup.empty:
                    plt.plot(warmup['run_number'], warmup['value'],
                            linestyle='--', marker='o', alpha=0.5, label=f'{short_name} (warmup)')

                if not counted.empty:
                    plt.plot(counted['run_number'], counted['value'],
                            linestyle='-', marker='o', label=short_name)

            plt.ylabel('Latency (ms)')
            plt.xlabel('Run Number')
            plt.title('Latency Trend Over Runs')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            png_path = output_dir / f"{filename}.png"
            plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close()
            files.append(png_path)
            logger.info(f"Created static latency trend: {png_path}")

        return files

    def _create_output_vs_latency_scatter(
        self, metrics_df: pd.DataFrame, output_dir: Path, filename: str
    ) -> List[Path]:
        """Create scatter plot of output length vs latency."""
        files = []

        # Need both latency and output_length for same runs
        latency_df = metrics_df[
            (metrics_df['metric_name'] == 'latency_ms') &
            (metrics_df['is_warmup'] == False)
        ]
        output_df = metrics_df[
            (metrics_df['metric_name'] == 'output_length') &
            (metrics_df['is_warmup'] == False)
        ]

        if latency_df.empty or output_df.empty:
            return files

        # Merge on model_id, endpoint, run_number
        merged = pd.merge(
            latency_df[['model_id', 'endpoint', 'run_number', 'value']],
            output_df[['model_id', 'endpoint', 'run_number', 'value']],
            on=['model_id', 'endpoint', 'run_number'],
            suffixes=('_latency', '_output')
        )

        if merged.empty:
            return files

        # Plotly version
        if self.plotly_available:
            fig = go.Figure()

            for model_id in merged['model_id'].unique():
                model_data = merged[merged['model_id'] == model_id]

                fig.add_trace(go.Scatter(
                    x=model_data['value_output'],
                    y=model_data['value_latency'],
                    name=self._shorten_model_name(model_id),
                    mode='markers',
                    marker=dict(size=10)
                ))

            fig.update_layout(
                title="Output Length vs Latency Correlation",
                xaxis_title="Output Length (tokens)",
                yaxis_title="Latency (ms)",
                height=500
            )

            html_path = output_dir / f"{filename}.html"
            fig.write_html(str(html_path))
            files.append(html_path)
            logger.info(f"Created correlation scatter: {html_path}")

        # Matplotlib version
        if self.matplotlib_available:
            plt.figure(figsize=(10, 6))

            for model_id in merged['model_id'].unique():
                model_data = merged[merged['model_id'] == model_id]

                plt.scatter(model_data['value_output'], model_data['value_latency'],
                           label=self._shorten_model_name(model_id), s=50, alpha=0.6)

            plt.xlabel('Output Length (tokens)')
            plt.ylabel('Latency (ms)')
            plt.title('Output Length vs Latency Correlation')
            plt.legend()
            plt.tight_layout()

            png_path = output_dir / f"{filename}.png"
            plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close()
            files.append(png_path)
            logger.info(f"Created static correlation scatter: {png_path}")

        return files

    def _create_percentiles_bar(
        self, aggregates_df: pd.DataFrame, output_dir: Path, filename: str
    ) -> List[Path]:
        """Create bar chart with p50, p95, p99 percentiles."""
        files = []

        latency_agg = aggregates_df[aggregates_df['metric_name'] == 'latency_ms']

        if latency_agg.empty:
            return files

        # Plotly version
        if self.plotly_available:
            model_ids = latency_agg['model_id'].unique()

            p50_values = [latency_agg[latency_agg['model_id'] == mid]['p50'].values[0] for mid in model_ids]
            p95_values = [latency_agg[latency_agg['model_id'] == mid]['p95'].values[0] for mid in model_ids]
            p99_values = [latency_agg[latency_agg['model_id'] == mid]['p99'].values[0] for mid in model_ids]

            x = [self._shorten_model_name(mid) for mid in model_ids]

            fig = go.Figure()
            fig.add_trace(go.Bar(name='p50 (median)', x=x, y=p50_values))
            fig.add_trace(go.Bar(name='p95', x=x, y=p95_values))
            fig.add_trace(go.Bar(name='p99', x=x, y=p99_values))

            fig.update_layout(
                title="Latency Percentiles (SLA Guarantees)",
                yaxis_title="Latency (ms)",
                xaxis_title="Model",
                barmode='group',
                height=500
            )

            html_path = output_dir / f"{filename}.html"
            fig.write_html(str(html_path))
            files.append(html_path)
            logger.info(f"Created percentiles chart: {html_path}")

        # Matplotlib version
        if self.matplotlib_available:
            model_ids = latency_agg['model_id'].unique()

            p50_values = np.array([latency_agg[latency_agg['model_id'] == mid]['p50'].values[0] for mid in model_ids])
            p95_values = np.array([latency_agg[latency_agg['model_id'] == mid]['p95'].values[0] for mid in model_ids])
            p99_values = np.array([latency_agg[latency_agg['model_id'] == mid]['p99'].values[0] for mid in model_ids])

            x = np.arange(len(model_ids))
            width = 0.25

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width, p50_values, width, label='p50 (median)')
            ax.bar(x, p95_values, width, label='p95')
            ax.bar(x + width, p99_values, width, label='p99')

            ax.set_ylabel('Latency (ms)')
            ax.set_xlabel('Model')
            ax.set_title('Latency Percentiles (SLA Guarantees)')
            ax.set_xticks(x)
            ax.set_xticklabels([self._shorten_model_name(mid) for mid in model_ids], rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()

            png_path = output_dir / f"{filename}.png"
            plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close()
            files.append(png_path)
            logger.info(f"Created static percentiles chart: {png_path}")

        return files

    def _create_combined_dashboard(
        self, metrics_df: pd.DataFrame, aggregates_df: pd.DataFrame,
        output_dir: Path, filename: str
    ) -> List[Path]:
        """Create multi-panel dashboard for Complete suite."""
        files = []

        # Only create Plotly dashboard (matplotlib multi-panel is complex)
        if not self.plotly_available:
            logger.info("Plotly not available - skipping combined dashboard")
            return files

        if metrics_df is None or aggregates_df is None:
            logger.warning("No data for combined dashboard")
            return files

        # Create 2x3 subplot dashboard (Speed + Resources overview)
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "Latency Comparison", "Memory Usage", "Resource Efficiency",
                "Latency Over Time", "CPU Usage", "Output Correlation"
            ),
            specs=[
                [{"type": "box"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Get per-model aggregates
        per_model = aggregates_df[
            (aggregates_df['aggregation_level'] == 'per_model') &
            (aggregates_df['aggregation_type'] == 'counted')
        ]

        # Panel 1: Latency Box Plot (row=1, col=1)
        latency_df = metrics_df[
            (metrics_df['metric_name'] == 'latency_ms') &
            (metrics_df['is_warmup'] == False)
        ]
        if not latency_df.empty:
            for model_id in latency_df['model_id'].unique():
                model_data = latency_df[latency_df['model_id'] == model_id]
                fig.add_trace(
                    go.Box(
                        y=model_data['value'],
                        name=self._shorten_model_name(model_id),
                        showlegend=False,
                        boxmean='sd'
                    ),
                    row=1, col=1
                )

        # Panel 2: Memory Usage Bar (row=1, col=2)
        memory_avg = per_model[per_model['metric_name'].str.contains('memory_mb_avg', na=False)]
        if not memory_avg.empty:
            model_ids = memory_avg['model_id'].unique()
            mem_values = [memory_avg[memory_avg['model_id'] == mid]['mean'].values[0] for mid in model_ids]
            fig.add_trace(
                go.Bar(
                    x=[self._shorten_model_name(mid) for mid in model_ids],
                    y=mem_values,
                    showlegend=False,
                    marker_color='steelblue'
                ),
                row=1, col=2
            )

        # Panel 3: Resource Efficiency Scatter (row=1, col=3)
        tokens_cpu = per_model[per_model['metric_name'] == 'tokens_per_cpu_percent']
        tokens_mem = per_model[per_model['metric_name'] == 'tokens_per_mb']
        if not tokens_cpu.empty and not tokens_mem.empty:
            for model_id in tokens_cpu['model_id'].unique():
                cpu_eff = tokens_cpu[tokens_cpu['model_id'] == model_id]['mean'].values
                mem_eff = tokens_mem[tokens_mem['model_id'] == model_id]['mean'].values
                if len(cpu_eff) > 0 and len(mem_eff) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[cpu_eff[0]],
                            y=[mem_eff[0]],
                            name=self._shorten_model_name(model_id),
                            mode='markers',
                            marker=dict(size=12),
                            showlegend=False
                        ),
                        row=1, col=3
                    )

        # Panel 4: Latency Over Time (row=2, col=1)
        if not latency_df.empty:
            for model_id in latency_df['model_id'].unique():
                model_data = latency_df[latency_df['model_id'] == model_id].sort_values('run_number')
                fig.add_trace(
                    go.Scatter(
                        x=model_data['run_number'],
                        y=model_data['value'],
                        name=self._shorten_model_name(model_id),
                        mode='lines+markers',
                        showlegend=True
                    ),
                    row=2, col=1
                )

        # Panel 5: CPU Usage Bar (row=2, col=2)
        cpu_avg = per_model[per_model['metric_name'].str.contains('cpu_percent_avg', na=False)]
        if not cpu_avg.empty:
            model_ids = cpu_avg['model_id'].unique()
            cpu_values = [cpu_avg[cpu_avg['model_id'] == mid]['mean'].values[0] for mid in model_ids]
            fig.add_trace(
                go.Bar(
                    x=[self._shorten_model_name(mid) for mid in model_ids],
                    y=cpu_values,
                    showlegend=False,
                    marker_color='indianred'
                ),
                row=2, col=2
            )

        # Panel 6: Output vs Latency Scatter (row=2, col=3)
        output_df = metrics_df[
            (metrics_df['metric_name'] == 'output_length') &
            (metrics_df['is_warmup'] == False)
        ]
        if not latency_df.empty and not output_df.empty:
            merged = pd.merge(
                latency_df[['model_id', 'endpoint', 'run_number', 'value']],
                output_df[['model_id', 'endpoint', 'run_number', 'value']],
                on=['model_id', 'endpoint', 'run_number'],
                suffixes=('_latency', '_output')
            )
            if not merged.empty:
                for model_id in merged['model_id'].unique():
                    model_data = merged[merged['model_id'] == model_id]
                    fig.add_trace(
                        go.Scatter(
                            x=model_data['value_output'],
                            y=model_data['value_latency'],
                            name=self._shorten_model_name(model_id),
                            mode='markers',
                            marker=dict(size=8),
                            showlegend=False
                        ),
                        row=2, col=3
                    )

        # Update axis labels
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
        fig.update_xaxes(title_text="Tokens/CPU%", row=1, col=3)
        fig.update_yaxes(title_text="Tokens/MB", row=1, col=3)
        fig.update_xaxes(title_text="Run #", row=2, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)
        fig.update_yaxes(title_text="CPU (%)", row=2, col=2)
        fig.update_xaxes(title_text="Output Length", row=2, col=3)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=3)

        fig.update_layout(
            title_text="Complete Benchmark Dashboard - Speed & Resources Overview",
            height=900,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            )
        )

        html_path = output_dir / f"{filename}.html"
        fig.write_html(str(html_path))
        files.append(html_path)
        logger.info(f"Created combined dashboard: {html_path}")

        return files

    def _create_resource_utilization_bars(
        self, aggregates_df: pd.DataFrame, output_dir: Path, filename: str
    ) -> List[Path]:
        """Create grouped bar chart for CPU and Memory utilization."""
        files = []

        # Look for CPU and memory metrics
        cpu_data = aggregates_df[aggregates_df['metric_name'].str.contains('cpu_percent', na=False)]
        memory_data = aggregates_df[aggregates_df['metric_name'].str.contains('memory_mb', na=False)]

        if cpu_data.empty and memory_data.empty:
            logger.warning("No CPU/Memory data available for utilization chart")
            return files

        # Plotly version
        if self.plotly_available:
            fig = go.Figure()

            if not cpu_data.empty:
                # Get average CPU usage per model
                cpu_avg = cpu_data[cpu_data['metric_name'].str.contains('avg', na=False)]
                if not cpu_avg.empty:
                    model_ids = cpu_avg['model_id'].unique()
                    cpu_values = [cpu_avg[cpu_avg['model_id'] == mid]['mean'].values[0] for mid in model_ids]

                    fig.add_trace(go.Bar(
                        name='CPU Usage (%)',
                        x=[self._shorten_model_name(mid) for mid in model_ids],
                        y=cpu_values,
                        marker_color='indianred'
                    ))

            if not memory_data.empty:
                # Get average memory usage per model
                memory_avg = memory_data[memory_data['metric_name'].str.contains('avg', na=False)]
                if not memory_avg.empty:
                    model_ids = memory_avg['model_id'].unique()
                    memory_values = [memory_avg[memory_avg['model_id'] == mid]['mean'].values[0] for mid in model_ids]

                    fig.add_trace(go.Bar(
                        name='Memory Usage (MB)',
                        x=[self._shorten_model_name(mid) for mid in model_ids],
                        y=memory_values,
                        marker_color='steelblue',
                        yaxis='y2'
                    ))

            # Create dual y-axis layout
            fig.update_layout(
                title="Resource Utilization per Model",
                xaxis_title="Model",
                yaxis=dict(title="CPU Usage (%)"),
                yaxis2=dict(title="Memory Usage (MB)", overlaying='y', side='right'),
                barmode='group',
                height=500
            )

            html_path = output_dir / f"{filename}.html"
            fig.write_html(str(html_path))
            files.append(html_path)
            logger.info(f"Created resource utilization chart: {html_path}")

        # Matplotlib version
        if self.matplotlib_available:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            x_labels = []
            cpu_values = []
            memory_values = []

            if not cpu_data.empty:
                cpu_avg = cpu_data[cpu_data['metric_name'].str.contains('avg', na=False)]
                if not cpu_avg.empty:
                    model_ids = cpu_avg['model_id'].unique()
                    x_labels = [self._shorten_model_name(mid) for mid in model_ids]
                    cpu_values = [cpu_avg[cpu_avg['model_id'] == mid]['mean'].values[0] for mid in model_ids]

            if not memory_data.empty:
                memory_avg = memory_data[memory_data['metric_name'].str.contains('avg', na=False)]
                if not memory_avg.empty:
                    model_ids = memory_avg['model_id'].unique()
                    if not x_labels:  # If CPU wasn't available, use memory model IDs
                        x_labels = [self._shorten_model_name(mid) for mid in model_ids]
                    memory_values = [memory_avg[memory_avg['model_id'] == mid]['mean'].values[0] for mid in model_ids]

            x = np.arange(len(x_labels))
            width = 0.35

            if cpu_values:
                ax1.bar(x - width/2, cpu_values, width, label='CPU %', color='indianred')

            ax1.set_xlabel('Model')
            ax1.set_ylabel('CPU Usage (%)', color='indianred')
            ax1.tick_params(axis='y', labelcolor='indianred')
            ax1.set_xticks(x)
            ax1.set_xticklabels(x_labels, rotation=45, ha='right')

            if memory_values:
                ax2 = ax1.twinx()
                ax2.bar(x + width/2, memory_values, width, label='Memory MB', color='steelblue')
                ax2.set_ylabel('Memory Usage (MB)', color='steelblue')
                ax2.tick_params(axis='y', labelcolor='steelblue')

            plt.title('Resource Utilization per Model')
            fig.tight_layout()

            png_path = output_dir / f"{filename}.png"
            plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close()
            files.append(png_path)
            logger.info(f"Created static resource utilization: {png_path}")

        return files

    def _create_resource_efficiency_scatter(
        self, aggregates_df: pd.DataFrame, output_dir: Path, filename: str
    ) -> List[Path]:
        """Create scatter plot showing resource efficiency metrics."""
        files = []

        # Look for efficiency metrics
        tokens_per_cpu = aggregates_df[aggregates_df['metric_name'] == 'tokens_per_cpu_percent']
        tokens_per_mb = aggregates_df[aggregates_df['metric_name'] == 'tokens_per_mb']

        if tokens_per_cpu.empty or tokens_per_mb.empty:
            logger.warning("No efficiency metrics available")
            return files

        # Plotly version
        if self.plotly_available:
            fig = go.Figure()

            model_ids = tokens_per_cpu['model_id'].unique()

            for model_id in model_ids:
                cpu_eff = tokens_per_cpu[tokens_per_cpu['model_id'] == model_id]['mean'].values
                mem_eff = tokens_per_mb[tokens_per_mb['model_id'] == model_id]['mean'].values

                if len(cpu_eff) > 0 and len(mem_eff) > 0:
                    fig.add_trace(go.Scatter(
                        x=[cpu_eff[0]],
                        y=[mem_eff[0]],
                        name=self._shorten_model_name(model_id),
                        mode='markers',
                        marker=dict(size=15)
                    ))

            fig.update_layout(
                title="Resource Efficiency (Higher = Better)",
                xaxis_title="Tokens per CPU %",
                yaxis_title="Tokens per MB Memory",
                height=500
            )

            html_path = output_dir / f"{filename}.html"
            fig.write_html(str(html_path))
            files.append(html_path)
            logger.info(f"Created efficiency scatter: {html_path}")

        # Matplotlib version
        if self.matplotlib_available:
            plt.figure(figsize=(10, 6))

            model_ids = tokens_per_cpu['model_id'].unique()

            for model_id in model_ids:
                cpu_eff = tokens_per_cpu[tokens_per_cpu['model_id'] == model_id]['mean'].values
                mem_eff = tokens_per_mb[tokens_per_mb['model_id'] == model_id]['mean'].values

                if len(cpu_eff) > 0 and len(mem_eff) > 0:
                    plt.scatter(cpu_eff[0], mem_eff[0],
                               label=self._shorten_model_name(model_id), s=100, alpha=0.7)

            plt.xlabel('Tokens per CPU %')
            plt.ylabel('Tokens per MB Memory')
            plt.title('Resource Efficiency (Higher = Better)')
            plt.legend()
            plt.tight_layout()

            png_path = output_dir / f"{filename}.png"
            plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close()
            files.append(png_path)
            logger.info(f"Created static efficiency scatter: {png_path}")

        return files

    def _create_resource_over_time(
        self, metrics_df: pd.DataFrame, output_dir: Path, filename: str
    ) -> List[Path]:
        """Create line chart showing resource usage over run numbers."""
        files = []

        # Look for CPU and memory metrics over time
        cpu_df = metrics_df[
            (metrics_df['metric_name'].str.contains('cpu_percent', na=False)) &
            (metrics_df['is_warmup'] == False)
        ]
        memory_df = metrics_df[
            (metrics_df['metric_name'].str.contains('memory_mb', na=False)) &
            (metrics_df['is_warmup'] == False)
        ]

        if cpu_df.empty and memory_df.empty:
            logger.warning("No resource timeline data available")
            return files

        # Plotly version
        if self.plotly_available:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # CPU usage over time
            if not cpu_df.empty:
                cpu_avg = cpu_df[cpu_df['metric_name'].str.contains('avg', na=False)]
                for model_id in cpu_avg['model_id'].unique():
                    model_data = cpu_avg[cpu_avg['model_id'] == model_id].sort_values('run_number')

                    fig.add_trace(
                        go.Scatter(
                            x=model_data['run_number'],
                            y=model_data['value'],
                            name=f'{self._shorten_model_name(model_id)} CPU',
                            mode='lines+markers',
                            line=dict(width=2)
                        ),
                        secondary_y=False
                    )

            # Memory usage over time
            if not memory_df.empty:
                memory_avg = memory_df[memory_df['metric_name'].str.contains('avg', na=False)]
                for model_id in memory_avg['model_id'].unique():
                    model_data = memory_avg[memory_avg['model_id'] == model_id].sort_values('run_number')

                    fig.add_trace(
                        go.Scatter(
                            x=model_data['run_number'],
                            y=model_data['value'],
                            name=f'{self._shorten_model_name(model_id)} Mem',
                            mode='lines+markers',
                            line=dict(dash='dash', width=2)
                        ),
                        secondary_y=True
                    )

            fig.update_xaxes(title_text="Run Number")
            fig.update_yaxes(title_text="CPU Usage (%)", secondary_y=False)
            fig.update_yaxes(title_text="Memory Usage (MB)", secondary_y=True)

            fig.update_layout(
                title="Resource Usage Over Time",
                height=500
            )

            html_path = output_dir / f"{filename}.html"
            fig.write_html(str(html_path))
            files.append(html_path)
            logger.info(f"Created resource timeline: {html_path}")

        # Matplotlib version
        if self.matplotlib_available:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # CPU on left axis
            if not cpu_df.empty:
                cpu_avg = cpu_df[cpu_df['metric_name'].str.contains('avg', na=False)]
                for model_id in cpu_avg['model_id'].unique():
                    model_data = cpu_avg[cpu_avg['model_id'] == model_id].sort_values('run_number')
                    ax1.plot(model_data['run_number'], model_data['value'],
                            marker='o', label=f'{self._shorten_model_name(model_id)} CPU')

            ax1.set_xlabel('Run Number')
            ax1.set_ylabel('CPU Usage (%)', color='tab:red')
            ax1.tick_params(axis='y', labelcolor='tab:red')

            # Memory on right axis
            if not memory_df.empty:
                ax2 = ax1.twinx()
                memory_avg = memory_df[memory_df['metric_name'].str.contains('avg', na=False)]
                for model_id in memory_avg['model_id'].unique():
                    model_data = memory_avg[memory_avg['model_id'] == model_id].sort_values('run_number')
                    ax2.plot(model_data['run_number'], model_data['value'],
                            marker='s', linestyle='--', label=f'{self._shorten_model_name(model_id)} Mem')

                ax2.set_ylabel('Memory Usage (MB)', color='tab:blue')
                ax2.tick_params(axis='y', labelcolor='tab:blue')

            plt.title('Resource Usage Over Time')
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            fig.tight_layout()

            png_path = output_dir / f"{filename}.png"
            plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close()
            files.append(png_path)
            logger.info(f"Created static resource timeline: {png_path}")

        return files

    def _create_memory_comparison(
        self, aggregates_df: pd.DataFrame, output_dir: Path, filename: str
    ) -> List[Path]:
        """Create bar chart comparing peak and average memory usage."""
        files = []

        memory_peak = aggregates_df[aggregates_df['metric_name'].str.contains('memory_mb_peak', na=False)]
        memory_avg = aggregates_df[aggregates_df['metric_name'].str.contains('memory_mb_avg', na=False)]

        if memory_peak.empty and memory_avg.empty:
            logger.warning("No memory comparison data available")
            return files

        # Plotly version
        if self.plotly_available:
            fig = go.Figure()

            if not memory_avg.empty:
                model_ids = memory_avg['model_id'].unique()
                avg_values = [memory_avg[memory_avg['model_id'] == mid]['mean'].values[0] for mid in model_ids]

                fig.add_trace(go.Bar(
                    name='Average Memory',
                    x=[self._shorten_model_name(mid) for mid in model_ids],
                    y=avg_values,
                    marker_color='lightblue'
                ))

            if not memory_peak.empty:
                model_ids = memory_peak['model_id'].unique()
                peak_values = [memory_peak[memory_peak['model_id'] == mid]['mean'].values[0] for mid in model_ids]

                fig.add_trace(go.Bar(
                    name='Peak Memory',
                    x=[self._shorten_model_name(mid) for mid in model_ids],
                    y=peak_values,
                    marker_color='darkblue'
                ))

            fig.update_layout(
                title="Memory Consumption Comparison",
                yaxis_title="Memory (MB)",
                xaxis_title="Model",
                barmode='group',
                height=500
            )

            html_path = output_dir / f"{filename}.html"
            fig.write_html(str(html_path))
            files.append(html_path)
            logger.info(f"Created memory comparison: {html_path}")

        # Matplotlib version
        if self.matplotlib_available:
            plt.figure(figsize=(10, 6))

            model_ids = []
            avg_values = []
            peak_values = []

            if not memory_avg.empty:
                model_ids = memory_avg['model_id'].unique()
                avg_values = [memory_avg[memory_avg['model_id'] == mid]['mean'].values[0] for mid in model_ids]

            if not memory_peak.empty:
                if not model_ids.any():
                    model_ids = memory_peak['model_id'].unique()
                peak_values = [memory_peak[memory_peak['model_id'] == mid]['mean'].values[0] for mid in model_ids]

            x = np.arange(len(model_ids))
            width = 0.35

            if avg_values:
                plt.bar(x - width/2, avg_values, width, label='Average Memory', color='lightblue')
            if peak_values:
                plt.bar(x + width/2, peak_values, width, label='Peak Memory', color='darkblue')

            plt.ylabel('Memory (MB)')
            plt.xlabel('Model')
            plt.title('Memory Consumption Comparison')
            plt.xticks(x, [self._shorten_model_name(mid) for mid in model_ids], rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()

            png_path = output_dir / f"{filename}.png"
            plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close()
            files.append(png_path)
            logger.info(f"Created static memory comparison: {png_path}")

        return files

    def _shorten_model_name(self, model_id: str) -> str:
        """Shorten long model names for display."""
        # Extract meaningful parts
        if "quantized:gguf:" in model_id:
            # Extract just filename
            parts = model_id.split("/")
            if parts:
                filename = parts[-1]
                # Remove .gguf extension
                return filename.replace(".gguf", "")

        return model_id

    def __repr__(self) -> str:
        """String representation."""
        libs = []
        if self.plotly_available:
            libs.append("Plotly")
        if self.matplotlib_available:
            libs.append("Matplotlib")

        return f"GraphGenerator(libraries={', '.join(libs) if libs else 'none'})"

"""
HTML Reporter with interactive Plotly charts.

Generates comprehensive HTML reports with:
- Executive summary
- Interactive Plotly visualizations
- Detailed metrics tables
- Model comparison views
- System information
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not installed - HTML reports will have limited functionality")

from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.metric_types import SuiteType


class HTMLReporter:
    """
    Generates interactive HTML reports with Plotly charts.

    Features:
    - Executive summary with key metrics
    - Interactive Plotly charts (latency, throughput, memory, etc.)
    - Detailed metrics tables
    - Multi-model comparison views
    - System information section
    - Responsive design
    """

    def __init__(self):
        """Initialize HTML reporter."""
        self.plotly_available = PLOTLY_AVAILABLE

        if not self.plotly_available:
            logger.warning("Plotly not available - HTML reports will be basic")

    def generate_html_report(
        self,
        result: SuiteResult,
        output_path: Path,
        metrics_df: Optional[pd.DataFrame] = None,
        aggregates_df: Optional[pd.DataFrame] = None
    ) -> Path:
        """
        Generate comprehensive HTML report with interactive charts.

        Args:
            result: SuiteResult to report
            output_path: Path to save HTML file
            metrics_df: Pre-loaded metrics DataFrame (optional)
            aggregates_df: Pre-loaded aggregates DataFrame (optional)

        Returns:
            Path to generated HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating HTML report: {output_path}")

        # Load CSVs if not provided
        if metrics_df is None:
            metrics_path = output_path.parent / "result_metrics.csv"
            if metrics_path.exists():
                metrics_df = pd.read_csv(metrics_path)

        if aggregates_df is None:
            aggregates_path = output_path.parent / "result_aggregates.csv"
            if aggregates_path.exists():
                aggregates_df = pd.read_csv(aggregates_path)

        # Generate HTML content
        html_content = self._build_html(result, metrics_df, aggregates_df)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report saved: {output_path}")
        return output_path

    def _build_html(
        self,
        result: SuiteResult,
        metrics_df: Optional[pd.DataFrame],
        aggregates_df: Optional[pd.DataFrame]
    ) -> str:
        """
        Build complete HTML document.

        Args:
            result: SuiteResult
            metrics_df: Metrics DataFrame
            aggregates_df: Aggregates DataFrame

        Returns:
            HTML content as string
        """
        # Build sections
        header = self._build_header(result)
        summary = self._build_summary_section(result, aggregates_df)
        charts = self._build_charts_section(result, metrics_df, aggregates_df)
        metrics_table = self._build_metrics_table(aggregates_df)
        system_info = self._build_system_info(result)
        footer = self._build_footer()

        # Combine into full HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Report - {result.suite_type.display_name()}</title>
    <style>
        {self._get_css()}
    </style>
    {"<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>" if self.plotly_available else ""}
</head>
<body>
    <div class="container">
        {header}
        {summary}
        {charts}
        {metrics_table}
        {system_info}
        {footer}
    </div>
</body>
</html>"""

        return html

    def _build_header(self, result: SuiteResult) -> str:
        """Build report header."""
        return f"""
        <header class="report-header">
            <h1>Benchmark Report</h1>
            <h2>{result.suite_type.display_name()}</h2>
            <div class="header-meta">
                <span class="badge badge-{result.status.value}">{result.status.value.upper()}</span>
                <span class="timestamp">{result.start_time.strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
        </header>
        """

    def _build_summary_section(
        self,
        result: SuiteResult,
        aggregates_df: Optional[pd.DataFrame]
    ) -> str:
        """Build executive summary section."""
        # Calculate key metrics
        duration_min = result.duration_seconds / 60
        success_rate = (result.successful_runs / result.total_runs * 100) if result.total_runs > 0 else 0

        # Model count
        model_count = len(result.models_tested)

        html = f"""
        <section class="summary-section">
            <h3>Executive Summary</h3>
            <div class="summary-cards">
                <div class="summary-card">
                    <div class="card-title">Duration</div>
                    <div class="card-value">{duration_min:.1f} min</div>
                    <div class="card-subtitle">{result.duration_seconds:.1f}s total</div>
                </div>
                <div class="summary-card">
                    <div class="card-title">Models Tested</div>
                    <div class="card-value">{model_count}</div>
                    <div class="card-subtitle">{', '.join(result.models_tested[:2])}{' ...' if model_count > 2 else ''}</div>
                </div>
                <div class="summary-card">
                    <div class="card-title">Total Runs</div>
                    <div class="card-value">{result.total_runs}</div>
                    <div class="card-subtitle">{result.successful_runs} successful</div>
                </div>
                <div class="summary-card">
                    <div class="card-title">Success Rate</div>
                    <div class="card-value">{success_rate:.1f}%</div>
                    <div class="card-subtitle">{'✓' if success_rate > 95 else '⚠'} {result.failed_runs} failed</div>
                </div>
            </div>
        """

        # Add suite-specific highlights
        if aggregates_df is not None and not aggregates_df.empty:
            html += self._build_suite_highlights(result.suite_type, aggregates_df)

        html += "</section>"
        return html

    def _build_suite_highlights(self, suite_type: SuiteType, aggregates_df: pd.DataFrame) -> str:
        """Build suite-specific metric highlights."""
        html = '<div class="highlights">'

        if suite_type == SuiteType.SPEED:
            # Show fastest model, avg latency, throughput
            if 'latency_ms_mean' in aggregates_df.columns:
                fastest_idx = aggregates_df['latency_ms_mean'].idxmin()
                fastest_model = aggregates_df.loc[fastest_idx, 'model_id']
                fastest_latency = aggregates_df.loc[fastest_idx, 'latency_ms_mean']

                html += f'<div class="highlight-item">⚡ Fastest: <strong>{fastest_model}</strong> ({fastest_latency:.1f}ms)</div>'

        elif suite_type == SuiteType.RESOURCES:
            # Show most efficient model
            if 'memory_mb_peak_mean' in aggregates_df.columns:
                efficient_idx = aggregates_df['memory_mb_peak_mean'].idxmin()
                efficient_model = aggregates_df.loc[efficient_idx, 'model_id']
                memory_usage = aggregates_df.loc[efficient_idx, 'memory_mb_peak_mean']

                html += f'<div class="highlight-item">💾 Most Efficient: <strong>{efficient_model}</strong> ({memory_usage:.0f}MB peak)</div>'

        html += '</div>'
        return html

    def _build_charts_section(
        self,
        result: SuiteResult,
        metrics_df: Optional[pd.DataFrame],
        aggregates_df: Optional[pd.DataFrame]
    ) -> str:
        """Build interactive charts section."""
        if not self.plotly_available or aggregates_df is None or aggregates_df.empty:
            return '<section class="charts-section"><p class="no-data">No charts available</p></section>'

        html = '<section class="charts-section"><h3>Interactive Visualizations</h3>'

        # Generate suite-specific charts
        if result.suite_type == SuiteType.SPEED:
            html += self._create_speed_charts(aggregates_df)
        elif result.suite_type == SuiteType.RESOURCES:
            html += self._create_resources_charts(aggregates_df)
        elif result.suite_type == SuiteType.QUALITY:
            html += self._create_quality_charts(aggregates_df)
        elif result.suite_type == SuiteType.STRESS:
            html += self._create_stress_charts(aggregates_df)
        elif result.suite_type == SuiteType.COMPLETE:
            html += self._create_complete_charts(aggregates_df)

        html += '</section>'
        return html

    def _create_speed_charts(self, df: pd.DataFrame) -> str:
        """Create charts for Speed suite."""
        charts_html = ""

        # Latency comparison
        if 'latency_ms_mean' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['model_id'],
                y=df['latency_ms_mean'],
                error_y=dict(type='data', array=df.get('latency_ms_std', [0]*len(df))),
                name='Mean Latency',
                marker_color='#3b82f6'
            ))
            fig.update_layout(
                title='Model Latency Comparison',
                xaxis_title='Model',
                yaxis_title='Latency (ms)',
                height=400
            )
            charts_html += f'<div class="chart-container">{fig.to_html(include_plotlyjs=False, div_id="latency-chart")}</div>'

        # Throughput comparison
        if 'throughput_tokens_per_sec_mean' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['model_id'],
                y=df['throughput_tokens_per_sec_mean'],
                name='Throughput',
                marker_color='#10b981'
            ))
            fig.update_layout(
                title='Model Throughput Comparison',
                xaxis_title='Model',
                yaxis_title='Tokens/sec',
                height=400
            )
            charts_html += f'<div class="chart-container">{fig.to_html(include_plotlyjs=False, div_id="throughput-chart")}</div>'

        return charts_html

    def _create_resources_charts(self, df: pd.DataFrame) -> str:
        """Create charts for Resources suite."""
        charts_html = ""

        # Memory usage
        if 'memory_mb_peak_mean' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['model_id'],
                y=df['memory_mb_peak_mean'],
                name='Peak Memory',
                marker_color='#f59e0b'
            ))
            fig.update_layout(
                title='Peak Memory Usage',
                xaxis_title='Model',
                yaxis_title='Memory (MB)',
                height=400
            )
            charts_html += f'<div class="chart-container">{fig.to_html(include_plotlyjs=False, div_id="memory-chart")}</div>'

        # CPU usage
        if 'cpu_percent_mean_mean' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['model_id'],
                y=df['cpu_percent_mean_mean'],
                name='Mean CPU',
                marker_color='#ef4444'
            ))
            fig.update_layout(
                title='CPU Utilization',
                xaxis_title='Model',
                yaxis_title='CPU Usage (%)',
                height=400
            )
            charts_html += f'<div class="chart-container">{fig.to_html(include_plotlyjs=False, div_id="cpu-chart")}</div>'

        return charts_html

    def _create_quality_charts(self, df: pd.DataFrame) -> str:
        """Create charts for Quality suite."""
        charts_html = ""

        if 'consistency_exact_match_percent_mean' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['model_id'],
                y=df['consistency_exact_match_percent_mean'],
                name='Exact Match %',
                marker_color='#8b5cf6'
            ))
            fig.update_layout(
                title='Output Consistency',
                xaxis_title='Model',
                yaxis_title='Exact Match (%)',
                height=400
            )
            charts_html += f'<div class="chart-container">{fig.to_html(include_plotlyjs=False, div_id="consistency-chart")}</div>'

        return charts_html

    def _create_stress_charts(self, df: pd.DataFrame) -> str:
        """Create charts for Stress suite."""
        charts_html = ""

        if 'latency_degradation_ms_mean' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['model_id'],
                y=df['latency_degradation_ms_mean'],
                name='Latency Degradation',
                marker_color='#ec4899'
            ))
            fig.update_layout(
                title='Performance Degradation',
                xaxis_title='Model',
                yaxis_title='Degradation (ms)',
                height=400
            )
            charts_html += f'<div class="chart-container">{fig.to_html(include_plotlyjs=False, div_id="degradation-chart")}</div>'

        return charts_html

    def _create_complete_charts(self, df: pd.DataFrame) -> str:
        """Create comprehensive charts for Complete suite."""
        # Show multiple metrics in a combined view
        return self._create_speed_charts(df) + self._create_resources_charts(df)

    def _build_metrics_table(self, aggregates_df: Optional[pd.DataFrame]) -> str:
        """Build detailed metrics table."""
        if aggregates_df is None or aggregates_df.empty:
            return '<section class="metrics-section"><p class="no-data">No metrics data available</p></section>'

        # Convert DataFrame to HTML table
        table_html = aggregates_df.to_html(
            classes=['metrics-table'],
            index=False,
            float_format=lambda x: f'{x:.2f}' if pd.notnull(x) else 'N/A'
        )

        return f"""
        <section class="metrics-section">
            <h3>Detailed Metrics</h3>
            <div class="table-container">
                {table_html}
            </div>
        </section>
        """

    def _build_system_info(self, result: SuiteResult) -> str:
        """Build system information section."""
        sys_info = result.system_info

        html = """
        <section class="system-section">
            <h3>System Information</h3>
            <table class="system-table">
        """

        # Add key system info
        for key, value in sys_info.items():
            if isinstance(value, dict):
                continue  # Skip nested dicts for now
            html += f"<tr><td class='sys-key'>{key}</td><td class='sys-value'>{value}</td></tr>"

        html += """
            </table>
        </section>
        """

        return html

    def _build_footer(self) -> str:
        """Build report footer."""
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        return f"""
        <footer class="report-footer">
            <p>Generated by Ekam-CLI at {now}</p>
            <p>Report version 1.0 | <a href="https://github.com/your-repo" target="_blank">Documentation</a></p>
        </footer>
        """

    def _get_css(self) -> str:
        """Get CSS styles for HTML report."""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f3f4f6;
            color: #1f2937;
            line-height: 1.6;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .report-header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .report-header h2 { font-size: 1.5em; opacity: 0.9; }
        .header-meta { margin-top: 20px; }

        .badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 600;
            margin-right: 15px;
        }
        .badge-success { background: #10b981; color: white; }
        .badge-partial { background: #f59e0b; color: white; }
        .badge-failed { background: #ef4444; color: white; }
        .timestamp { opacity: 0.8; }

        section {
            background: white;
            padding: 30px;
            margin-bottom: 25px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        section h3 { font-size: 1.75em; margin-bottom: 20px; color: #374151; }

        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .summary-card {
            background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }
        .card-title { font-size: 0.9em; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
        .card-value { font-size: 2.5em; font-weight: 700; color: #1f2937; margin: 10px 0; }
        .card-subtitle { font-size: 0.9em; color: #6b7280; }

        .highlights {
            margin-top: 25px;
            padding: 20px;
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            border-radius: 8px;
        }
        .highlight-item { margin: 10px 0; font-size: 1.1em; }

        .chart-container { margin: 30px 0; }

        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .metrics-table th { background: #f3f4f6; padding: 12px; text-align: left; font-weight: 600; }
        .metrics-table td { padding: 10px; border-bottom: 1px solid #e5e7eb; }
        .metrics-table tr:hover { background: #f9fafb; }

        .system-table {
            width: 100%;
            border-collapse: collapse;
        }
        .system-table td { padding: 10px; border-bottom: 1px solid #e5e7eb; }
        .sys-key { font-weight: 600; width: 30%; color: #6b7280; }
        .sys-value { color: #1f2937; }

        .report-footer {
            text-align: center;
            padding: 30px;
            color: #6b7280;
            font-size: 0.9em;
        }
        .report-footer a { color: #3b82f6; text-decoration: none; }
        .report-footer a:hover { text-decoration: underline; }

        .no-data { text-align: center; color: #9ca3af; padding: 40px; }

        @media (max-width: 768px) {
            .container { padding: 10px; }
            .report-header { padding: 25px; }
            .report-header h1 { font-size: 1.75em; }
            .summary-cards { grid-template-columns: 1fr; }
        }
        """


def generate_html_report(
    result: SuiteResult,
    output_path: Path,
    metrics_df: Optional[pd.DataFrame] = None,
    aggregates_df: Optional[pd.DataFrame] = None
) -> Path:
    """
    Convenience function to generate HTML report.

    Args:
        result: SuiteResult to report
        output_path: Path to save HTML file
        metrics_df: Pre-loaded metrics DataFrame (optional)
        aggregates_df: Pre-loaded aggregates DataFrame (optional)

    Returns:
        Path to generated HTML file
    """
    reporter = HTMLReporter()
    return reporter.generate_html_report(result, output_path, metrics_df, aggregates_df)

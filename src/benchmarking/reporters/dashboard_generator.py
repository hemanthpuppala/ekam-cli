"""
Dashboard HTML generator for benchmark results.

Generates interactive HTML dashboard from CSV and JSON result files.
Provides a Splunk-like visualization experience with dark/light themes.
"""

from pathlib import Path
import shutil
from typing import Optional
from loguru import logger

from src.benchmarking.models.suite_result import SuiteResult


class DashboardGenerator:
    """
    Generates interactive HTML dashboards for benchmark results.

    Features:
    - Self-contained HTML with embedded data loading via JavaScript
    - Dark/light theme toggle
    - KPI cards with suite-specific metrics
    - Interactive filters (model, endpoint)
    - Drill-down details on metrics
    - Advanced visualizations (heatmaps, distributions, trends)
    - Error visualization and reporting
    - Responsive design for desktop/mobile
    """

    def __init__(self):
        """Initialize dashboard generator."""
        # Get the path to the dashboard template
        self.template_path = Path(__file__).parent / "dashboard.html"

        if not self.template_path.exists():
            logger.error(f"Dashboard template not found at {self.template_path}")

    def generate_dashboard(
        self,
        result_dir: Path,
        result: Optional[SuiteResult] = None
    ) -> Optional[Path]:
        """
        Generate interactive HTML dashboard for a benchmark result.

        Embeds CSV and JSON data directly into the HTML file for offline viewing.

        Args:
            result_dir: Path to the benchmark results directory
            result: Optional SuiteResult for additional context

        Returns:
            Path to generated dashboard HTML file, or None if failed
        """
        result_dir = Path(result_dir)

        if not result_dir.exists():
            logger.error(f"Result directory does not exist: {result_dir}")
            return None

        # Destination for dashboard
        dashboard_path = result_dir / "dashboard.html"

        try:
            # Read template
            if not self.template_path.exists():
                logger.error(f"Template not found: {self.template_path}")
                return None

            with open(self.template_path, 'r', encoding='utf-8') as f:
                template_html = f.read()

            # Load data files
            json_data = self._load_json_data(result_dir)
            metrics_csv = self._load_csv_data(result_dir / "result_metrics.csv")
            aggregates_csv = self._load_csv_data(result_dir / "result_aggregates.csv")

            # Embed data into HTML
            embedded_html = self._embed_data_in_html(
                template_html,
                json_data,
                metrics_csv,
                aggregates_csv
            )

            # Write embedded HTML
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(embedded_html)

            logger.info(f"Dashboard generated with embedded data: {dashboard_path}")

            # Create a metadata file with dashboard info
            self._create_dashboard_metadata(result_dir, result)

            return dashboard_path

        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
            logger.exception(e)
            return None

    def _load_json_data(self, result_dir: Path) -> str:
        """Load result.json and return as escaped JSON string for embedding."""
        try:
            json_path = result_dir / "result.json"
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Escape for embedding in JavaScript template literal (backticks)
                # Must escape: \ (backslash), ` (backtick), $ (template expression)
                return content.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
        except Exception as e:
            logger.warning(f"Could not load result.json: {e}")
        return "{}"

    def _load_csv_data(self, csv_path: Path) -> str:
        """Load CSV file and return as escaped string."""
        try:
            if csv_path.exists():
                with open(csv_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Escape for embedding in JavaScript
                return content.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
        except Exception as e:
            logger.warning(f"Could not load {csv_path.name}: {e}")
        return ""

    def _embed_data_in_html(
        self,
        html_template: str,
        json_data: str,
        metrics_csv: str,
        aggregates_csv: str
    ) -> str:
        """
        Embed data, CSS, and JS into HTML template.

        Replaces placeholders and inlines external resources for a self-contained HTML file.
        """
        # Replace EMBEDDED_DATA placeholder in the template
        html_template = html_template.replace(
            'json: ``',
            f'json: `{json_data}`'
        ).replace(
            'metricsCSV: ``',
            f'metricsCSV: `{metrics_csv}`'
        ).replace(
            'aggregatesCSV: ``',
            f'aggregatesCSV: `{aggregates_csv}`'
        )

        # Inline CSS
        css_path = self.template_path.parent / "dashboard.css"
        if css_path.exists():
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
            html_template = html_template.replace(
                '<link rel="stylesheet" href="dashboard.css">',
                f'<style>\n{css_content}\n    </style>'
            )
        else:
            logger.warning(f"CSS file not found: {css_path}")

        # Inline Metric Metadata JavaScript
        metadata_js_path = self.template_path.parent / "metric_metadata.js"
        if metadata_js_path.exists():
            with open(metadata_js_path, 'r', encoding='utf-8') as f:
                metadata_content = f.read()
            html_template = html_template.replace(
                '<script src="metric_metadata.js"></script>',
                f'<script>\n{metadata_content}\n    </script>'
            )
        else:
            logger.warning(f"Metric metadata JavaScript file not found: {metadata_js_path}")

        # Inline Main JavaScript
        js_path = self.template_path.parent / "dashboard.js"
        if js_path.exists():
            with open(js_path, 'r', encoding='utf-8') as f:
                js_content = f.read()
            html_template = html_template.replace(
                '<script src="dashboard.js"></script>',
                f'<script>\n{js_content}\n    </script>'
            )
        else:
            logger.warning(f"JavaScript file not found: {js_path}")

        # Inline Enhanced Dashboard Features JavaScript
        enhancements_js_path = self.template_path.parent / "dashboard_enhancements.js"
        if enhancements_js_path.exists():
            with open(enhancements_js_path, 'r', encoding='utf-8') as f:
                enhancements_content = f.read()
            html_template = html_template.replace(
                '<script src="dashboard_enhancements.js"></script>',
                f'<script>\n{enhancements_content}\n    </script>'
            )
        else:
            logger.warning(f"Enhanced dashboard JavaScript file not found: {enhancements_js_path}")

        return html_template

    def _create_dashboard_metadata(
        self,
        result_dir: Path,
        result: Optional[SuiteResult] = None
    ) -> None:
        """
        Create dashboard metadata file for reference.

        Args:
            result_dir: Result directory path
            result: Optional SuiteResult for metadata
        """
        try:
            import json
            from datetime import datetime

            metadata = {
                "dashboard_generated": datetime.utcnow().isoformat(),
                "version": "1.0",
                "required_files": [
                    "result.json",
                    "result_metrics.csv",
                    "result_aggregates.csv"
                ],
                "suite_type": result.suite_type.value if result else "unknown",
                "model_type": result.model_type.value if result else "unknown",
                "note": "Open dashboard.html in a web browser to view the interactive dashboard"
            }

            metadata_path = result_dir / "dashboard_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not create dashboard metadata: {e}")

    @staticmethod
    def get_dashboard_opening_message(dashboard_path: Path) -> str:
        """
        Get formatted message for user to open dashboard.

        Args:
            dashboard_path: Path to the dashboard HTML file

        Returns:
            Formatted message string
        """
        absolute_path = dashboard_path.resolve()
        relative_path = dashboard_path.name

        message = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ✅ BENCHMARK SUITE COMPLETED                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 Interactive Dashboard Generated:
   File: {relative_path}
   Location: {dashboard_path.parent}

🌐 To view your results:
   1. Open this file in your web browser:
      {absolute_path}

   2. Or use command line:
      macOS/Linux: open "{absolute_path}"
      Windows: start "{absolute_path}"

✨ Dashboard Features:
   • KPI cards with key metrics
   • Dark/Light theme toggle
   • Interactive filters (model, endpoint)
   • Suite-specific visualizations
   • Detailed metrics tables
   • Error reporting
   • Responsive design

📈 Available Metrics:
   • result.json - Full benchmark data
   • result_metrics.csv - Individual metric observations
   • result_aggregates.csv - Aggregated statistics per model

════════════════════════════════════════════════════════════════════════════════
"""
        return message

    @staticmethod
    def show_dashboard_message(dashboard_path: Path) -> None:
        """
        Print formatted message for user to open dashboard.

        Args:
            dashboard_path: Path to the dashboard HTML file
        """
        from src.cli.tui_manager import tui

        message = DashboardGenerator.get_dashboard_opening_message(dashboard_path)

        # Use TUI manager for consistent styling
        try:
            tui.show_message(
                message,
                title="Dashboard Ready",
                style="green"
            )
        except Exception:
            # Fallback to direct printing if TUI not available
            print(message)

    @staticmethod
    def get_quick_dashboard_link(dashboard_path: Path) -> str:
        """
        Get a quick link/path for opening dashboard.

        Args:
            dashboard_path: Path to the dashboard HTML file

        Returns:
            One-liner instruction
        """
        absolute_path = dashboard_path.resolve()
        return f"📊 Open dashboard: {absolute_path}"


def generate_html_dashboard(
    result_dir: Path,
    result: Optional[SuiteResult] = None
) -> Optional[Path]:
    """
    Convenience function to generate dashboard.

    Args:
        result_dir: Path to benchmark results directory
        result: Optional SuiteResult for context

    Returns:
        Path to generated dashboard, or None if failed
    """
    generator = DashboardGenerator()
    return generator.generate_dashboard(result_dir, result)

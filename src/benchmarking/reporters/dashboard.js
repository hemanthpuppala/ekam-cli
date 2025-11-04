        // ===== GLOBAL STATE =====
        let metricsData = [];
        let aggregatesData = [];
        let filteredData = [];
        let allMetrics = new Set();
        let allModels = new Set();
        let pinnedRuns = [];
        let currentModalRun = null;
        let isDarkMode = true;
        let chartInstances = {};

        // ===== INITIALIZATION =====
        document.addEventListener('DOMContentLoaded', () => {
            loadData();
        });

        function loadData() {
            if (!window.EMBEDDED_DATA) {
                showAlert('No embedded data found!', 'warning');
                return;
            }

            try {
                // Parse metrics CSV
                if (window.EMBEDDED_DATA.metricsCSV) {
                    metricsData = parseCSV(window.EMBEDDED_DATA.metricsCSV);
                    console.log('Loaded metrics:', metricsData.length);
                } else {
                    showAlert('No metrics CSV data found', 'warning');
                }

                // Parse aggregates CSV
                if (window.EMBEDDED_DATA.aggregatesCSV) {
                    aggregatesData = parseCSV(window.EMBEDDED_DATA.aggregatesCSV);
                    console.log('Loaded aggregates:', aggregatesData.length);
                }

                // Check for missing columns
                checkDataQuality();

                // Extract unique values
                extractUniqueValues();

                // Initialize filters
                initializeFilters();

                // Apply initial filters
                filteredData = metricsData;
                applyFilters();

                // Render dashboard
                renderDashboard();

            } catch (error) {
                console.error('Error loading data:', error);
                showAlert('Error loading data: ' + error.message, 'warning');
            }
        }

        function parseCSV(csvString) {
            // Split CSV into lines, but respect quoted multi-line fields
            const lines = [];
            let currentLine = '';
            let inQuotes = false;

            for (let i = 0; i < csvString.length; i++) {
                const char = csvString[i];

                if (char === '"') {
                    inQuotes = !inQuotes;
                    currentLine += char;
                } else if (char === '\n' && !inQuotes) {
                    if (currentLine.trim()) {
                        lines.push(currentLine);
                    }
                    currentLine = '';
                } else {
                    currentLine += char;
                }
            }

            // Add the last line if it exists
            if (currentLine.trim()) {
                lines.push(currentLine);
            }

            if (lines.length < 2) return [];

            const headers = parseCSVLine(lines[0]);
            const data = [];

            for (let i = 1; i < lines.length; i++) {
                const values = parseCSVLine(lines[i]);
                if (values.length === headers.length) {
                    const row = {};
                    headers.forEach((header, index) => {
                        row[header] = values[index];
                    });
                    data.push(row);
                }
            }

            return data;
        }

        function parseCSVLine(line) {
            const result = [];
            let current = '';
            let inQuotes = false;

            for (let i = 0; i < line.length; i++) {
                const char = line[i];
                if (char === '"') {
                    inQuotes = !inQuotes;
                } else if (char === ',' && !inQuotes) {
                    result.push(current.trim());
                    current = '';
                } else {
                    current += char;
                }
            }
            result.push(current.trim());
            return result;
        }

        function checkDataQuality() {
            if (metricsData.length === 0) return;

            const hasPrompt = metricsData[0].hasOwnProperty('input_prompt');
            const hasResponse = metricsData[0].hasOwnProperty('raw_response');
            const hasImage = metricsData[0].hasOwnProperty('input_image_path');

            if (!hasPrompt || !hasResponse) {
                showAlert('⚠️ Missing input_prompt or raw_response columns in CSV data. Some features will be limited.', 'warning');
            }

            if (!hasImage) {
                showAlert('ℹ️ No input_image_path column found. Image previews disabled.', 'info');
            }
        }

        function extractUniqueValues() {
            metricsData.forEach(row => {
                if (row.metric_name) allMetrics.add(row.metric_name);
                if (row.model_id) allModels.add(row.model_id);
            });

            console.log('Unique metrics:', Array.from(allMetrics));
            console.log('Unique models:', Array.from(allModels));
        }

        function initializeFilters() {
            // Populate model filter
            const modelFilter = document.getElementById('model-filter');
            Array.from(allModels).sort().forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelFilter.appendChild(option);
            });

            // Populate metric filter
            const metricFilter = document.getElementById('metric-filter');
            Array.from(allMetrics).sort().forEach(metric => {
                const option = document.createElement('option');
                option.value = metric;
                option.textContent = metric;
                metricFilter.appendChild(option);
            });

        }

        function applyFilters() {
            const modelFilter = document.getElementById('model-filter').value;
            const metricFilter = document.getElementById('metric-filter').value;

            filteredData = metricsData.filter(row => {
                // Model filter
                if (modelFilter && row.model_id !== modelFilter) return false;

                // Metric filter
                if (metricFilter && row.metric_name !== metricFilter) return false;

                return true;
            });

            console.log('Filtered data:', filteredData.length);
            renderDashboard();
        }

        function renderDashboard() {
            renderStatsOverview();
            renderCharts();
        }

        function renderStatsOverview() {
            const container = document.getElementById('stats-overview');
            container.innerHTML = '';

            // Get currently selected metric filter
            const metricFilter = document.getElementById('metric-filter').value;

            // Calculate BASELINE: All unique runs across ALL metrics (reference)
            const allRunsBaseline = new Set(metricsData.map(r => {
                const sourceSuite = r.meta_source_suite || 'unknown';
                return `${r.model_id}|${r.endpoint}|${sourceSuite}|${r.run_number}`;
            })).size;

            // Calculate stats for FILTERED data
            // Count actual runs: unique (model_id, endpoint, source_suite, run_number) combinations
            // This accounts for run_number overlap between sub-suites (Speed run 1, Quality run 1, etc.)
            const totalRuns = new Set(filteredData.map(r => {
                const sourceSuite = r.meta_source_suite || 'unknown';
                return `${r.model_id}|${r.endpoint}|${sourceSuite}|${r.run_number}`;
            })).size;
            const totalModels = new Set(filteredData.map(r => r.model_id)).size;
            const totalMetrics = new Set(filteredData.map(r => r.metric_name)).size;

            // Check for missing data (REAL data, not mocked)
            const missingRunsInfo = calculateMissingRuns(metricFilter, filteredData, metricsData);

            // Average latency
            const latencies = filteredData
                .filter(r => r.metric_name === 'latency_s' || r.metric_name === 'latency')
                .map(r => parseFloat(r.value))
                .filter(v => !isNaN(v));
            const avgLatency = latencies.length > 0
                ? (latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(3)
                : 'N/A';

            const stats = [
                { label: 'Total Runs', value: totalRuns, unit: '', missingInfo: missingRunsInfo },
                { label: 'Models Tested', value: totalModels, unit: '' },
                { label: 'Metrics Tracked', value: totalMetrics, unit: '' },
                { label: 'Avg Latency', value: avgLatency, unit: 's' }
            ];

            stats.forEach(stat => {
                const card = document.createElement('div');
                card.className = 'stat-card';

                // Build warning indicator if data is missing
                let warningHTML = '';
                if (stat.missingInfo && stat.missingInfo.count > 0) {
                    warningHTML = `
                        <span class="missing-data-warning" data-missing-info='${JSON.stringify(stat.missingInfo)}'>
                            ⚠️
                            <span class="missing-data-tooltip">Missing data - Click for details</span>
                        </span>
                    `;
                }

                card.innerHTML = `
                    <div class="stat-label">${stat.label}</div>
                    <div class="stat-value">
                        ${stat.value} ${warningHTML}
                        <span class="stat-unit">${stat.unit}</span>
                    </div>
                `;
                container.appendChild(card);
            });

            // Add click handlers to missing data warnings
            document.querySelectorAll('.missing-data-warning').forEach(warning => {
                warning.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const missingInfo = JSON.parse(warning.getAttribute('data-missing-info'));
                    showMissingDataModal(missingInfo);
                });
            });
        }

        function showMissingDataModal(missingInfo) {
            const modal = document.getElementById('missing-data-modal');
            const modalContent = document.getElementById('missing-data-modal-content');

            // Build modal content
            let content = `
                <div class="modal-section">
                    <h3 style="color: var(--accent-warning); margin-bottom: 1rem;">⚠️ Missing Data Alert</h3>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                        <strong>${missingInfo.count}</strong> run(s) are missing data for the currently selected metric.
                    </p>
                    <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1.5rem;">
                        Baseline comparison: <strong>${missingInfo.total}</strong> total runs available from complete metrics.
                    </p>
                </div>

                <div class="modal-section">
                    <h4 style="color: var(--accent-primary); margin-bottom: 0.75rem;">Affected Runs by Model</h4>
                    <div class="missing-runs-list">
            `;

            for (const [model, runs] of Object.entries(missingInfo.missingByModel)) {
                runs.sort((a, b) => a - b);
                const runDisplay = runs.length > 10
                    ? `${runs.slice(0, 10).join(', ')}... <span style="color: var(--accent-warning);">(${runs.length} total)</span>`
                    : runs.join(', ');

                content += `
                    <div class="missing-run-item">
                        <div class="missing-run-model">${model}</div>
                        <div class="missing-run-numbers">Runs: ${runDisplay}</div>
                    </div>
                `;
            }

            content += `
                    </div>
                </div>

                <div class="modal-section">
                    <h4 style="color: var(--accent-info); margin-bottom: 0.75rem;">Likely Cause</h4>
                    <p style="color: var(--text-secondary); line-height: 1.6;">
                        ${missingInfo.reason || 'Data collection may have failed during these specific inference runs.'}
                    </p>
                    <div style="background: var(--bg-tertiary); border-left: 3px solid var(--accent-info); padding: 1rem; margin-top: 1rem; border-radius: 4px;">
                        <p style="color: var(--text-muted); font-size: 0.9rem; margin: 0;">
                            <strong>Note:</strong> Other metrics for these runs may have succeeded. This indicates a
                            specific collection failure for this particular metric during inference.
                        </p>
                    </div>
                </div>
            `;

            modalContent.innerHTML = content;
            modal.classList.add('active');
        }

        function calculateMissingRuns(metricFilter, filteredData, allData) {
            // If no metric filter selected, no missing data to report
            if (!metricFilter) {
                return null;
            }

            // Get all runs that should exist (from a complete metric like latency)
            const completeMetricRuns = new Set();
            const currentMetricRuns = new Set();

            // Build set of all runs per model+suite
            const runsByMetric = {};

            for (const row of allData) {
                const sourceSuite = row.meta_source_suite || 'unknown';
                const runKey = `${row.model_id}|${row.endpoint}|${sourceSuite}|${row.run_number}`;
                const metricName = row.metric_name;

                if (!runsByMetric[metricName]) {
                    runsByMetric[metricName] = new Set();
                }
                runsByMetric[metricName].add(runKey);
            }

            // Find the metric with most runs (baseline)
            let maxRuns = 0;
            let baselineMetric = null;
            for (const [metric, runs] of Object.entries(runsByMetric)) {
                if (runs.size > maxRuns) {
                    maxRuns = runs.size;
                    baselineMetric = metric;
                    completeMetricRuns.clear();
                    runs.forEach(r => completeMetricRuns.add(r));
                }
            }

            // Get runs for current metric
            if (runsByMetric[metricFilter]) {
                runsByMetric[metricFilter].forEach(r => currentMetricRuns.add(r));
            }

            // Calculate missing
            const missingRuns = [];
            for (const runKey of completeMetricRuns) {
                if (!currentMetricRuns.has(runKey)) {
                    missingRuns.push(runKey);
                }
            }

            if (missingRuns.length === 0) {
                return null;
            }

            // Group missing runs by model
            const missingByModel = {};
            for (const runKey of missingRuns) {
                const [modelId, endpoint, suite, runNum] = runKey.split('|');
                const modelName = modelId.split('/').pop();

                if (!missingByModel[modelName]) {
                    missingByModel[modelName] = [];
                }
                missingByModel[modelName].push(parseInt(runNum));
            }

            // Build detailed tooltip
            let tooltip = `⚠️ Missing Data Alert\n\n`;
            tooltip += `${missingRuns.length} run(s) missing "${metricFilter}" data\n`;
            tooltip += `(Baseline: ${maxRuns} runs from "${baselineMetric}")\n\n`;
            tooltip += `Affected Runs:\n`;

            for (const [model, runs] of Object.entries(missingByModel)) {
                runs.sort((a, b) => a - b);
                const runList = runs.length > 5
                    ? `${runs.slice(0, 5).join(', ')}... (${runs.length} total)`
                    : runs.join(', ');
                tooltip += `• ${model}: Runs #${runList}\n`;
            }

            // Determine likely reason for failure
            let reason = '';
            if (metricFilter.includes('cpu')) {
                reason = 'CPU monitoring failed due to system sampling issues. This can occur when inference completes faster than the CPU sampling interval, or when system-level CPU metrics are temporarily unavailable.';
            } else if (metricFilter.includes('memory')) {
                reason = 'Memory monitoring failed due to resource access issues. This may happen when memory metrics cannot be captured during rapid allocation/deallocation cycles.';
            } else if (metricFilter.includes('gpu')) {
                reason = 'GPU monitoring failed due to driver or access issues. GPU metrics may be unavailable if the device is under heavy load or driver communication fails.';
            } else {
                reason = 'Metric collection failed during inference. This could be due to timing issues, resource constraints, or specific edge cases during model execution.';
            }

            return {
                count: missingRuns.length,
                total: maxRuns,
                missingByModel: missingByModel,
                reason: reason,
                metricName: metricFilter
            };
        }

        function renderCharts() {
            const container = document.getElementById('dashboard-grid');
            container.innerHTML = '';

            if (filteredData.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">📊</div>
                        <div class="empty-state-text">No data to display. Try adjusting your filters.</div>
                    </div>
                `;
                return;
            }

            // Get all unique metrics
            const metrics = Array.from(new Set(filteredData.map(r => r.metric_name)));

            metrics.forEach(metric => {
                // Bar chart - model comparison
                renderBarChart(metric);

                // Time series - trend over runs
                renderTimeSeriesChart(metric);
            });

            // Heatmap - model vs metric
            renderHeatmap();

            // Box plot - statistical distribution
            metrics.forEach(metric => {
                renderBoxPlot(metric);
            });

            // Scatter plot - metric correlations
            if (metrics.length >= 2) {
                renderScatterPlot(metrics[0], metrics[1]);
            }
        }

        function renderBarChart(metric) {
            const container = document.getElementById('dashboard-grid');
            const card = createChartCard(`${metric} - Model Comparison`, 'bar-chart-' + metric);
            container.appendChild(card);

            const metricData = filteredData.filter(r => r.metric_name === metric);

            // Group by model and calculate average
            const modelGroups = {};
            metricData.forEach(row => {
                if (!modelGroups[row.model_id]) {
                    modelGroups[row.model_id] = [];
                }
                const value = parseFloat(row.value);
                if (!isNaN(value)) {
                    modelGroups[row.model_id].push(value);
                }
            });

            const models = Object.keys(modelGroups);
            const values = models.map(model => {
                const vals = modelGroups[model];
                return vals.reduce((a, b) => a + b, 0) / vals.length;
            });

            const trace = {
                x: models,
                y: values,
                type: 'bar',
                marker: {
                    color: values.map((_, i) => getChartColor(i)),
                    line: { width: 1, color: isDarkMode ? '#000' : '#fff' }
                },
                hovertemplate: '<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>'
            };

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: isDarkMode ? '#c3c7cc' : '#5c6470', size: 12 },
                margin: { t: 20, r: 20, b: 80, l: 60 },
                xaxis: {
                    title: 'Model',
                    gridcolor: isDarkMode ? '#3a3d42' : '#e5e9f0',
                    tickangle: -45
                },
                yaxis: {
                    title: metricData[0]?.unit || 'Value',
                    gridcolor: isDarkMode ? '#3a3d42' : '#e5e9f0'
                },
                hovermode: 'closest'
            };

            const config = { responsive: true, displayModeBar: true };

            Plotly.newPlot('bar-chart-' + metric, [trace], layout, config);

            // Note: Bar chart shows aggregated data (mean across runs)
            // Click events removed as aggregated values don't have individual run context
        }

        function renderTimeSeriesChart(metric) {
            const container = document.getElementById('dashboard-grid');
            const card = createChartCard(`${metric} - Trend Over Time`, 'timeseries-' + metric);
            container.appendChild(card);

            const metricData = filteredData.filter(r => r.metric_name === metric);

            // Group by model
            const modelGroups = {};
            metricData.forEach(row => {
                if (!modelGroups[row.model_id]) {
                    modelGroups[row.model_id] = [];
                }
                modelGroups[row.model_id].push(row);
            });

            const traces = [];
            Object.keys(modelGroups).forEach((model, idx) => {
                const data = modelGroups[model].sort((a, b) => {
                    const timeA = a.timestamp ? new Date(a.timestamp) : new Date(0);
                    const timeB = b.timestamp ? new Date(b.timestamp) : new Date(0);
                    return timeA - timeB;
                });

                traces.push({
                    x: data.map(r => r.timestamp || r.run_number),
                    y: data.map(r => parseFloat(r.value)),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: model,
                    line: { color: getChartColor(idx), width: 2 },
                    marker: { size: 6 },
                    hovertemplate: '<b>%{fullData.name}</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
                });
            });

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: isDarkMode ? '#c3c7cc' : '#5c6470', size: 12 },
                margin: { t: 20, r: 20, b: 60, l: 60 },
                xaxis: {
                    title: 'Time / Run',
                    gridcolor: isDarkMode ? '#3a3d42' : '#e5e9f0'
                },
                yaxis: {
                    title: metricData[0]?.unit || 'Value',
                    gridcolor: isDarkMode ? '#3a3d42' : '#e5e9f0'
                },
                legend: {
                    orientation: 'h',
                    y: -0.2
                },
                hovermode: 'closest'
            };

            const config = { responsive: true, displayModeBar: true };

            Plotly.newPlot('timeseries-' + metric, traces, layout, config);

            // Add click event to show individual run details
            document.getElementById('timeseries-' + metric).on('plotly_click', (data) => {
                const pointIndex = data.points[0].pointIndex;
                const traceIndex = data.points[0].curveNumber;
                const modelId = Object.keys(modelGroups)[traceIndex];
                const modelData = modelGroups[modelId].sort((a, b) => {
                    const timeA = a.timestamp ? new Date(a.timestamp) : new Date(0);
                    const timeB = b.timestamp ? new Date(b.timestamp) : new Date(0);
                    return timeA - timeB;
                });

                if (modelData[pointIndex]) {
                    showRunDetails(modelData[pointIndex]);
                }
            });
        }

        function renderHeatmap() {
            const container = document.getElementById('dashboard-grid');
            const card = createChartCard('Performance Heatmap - Model vs Metric', 'heatmap');
            container.appendChild(card);

            const models = Array.from(new Set(filteredData.map(r => r.model_id)));
            const metrics = Array.from(new Set(filteredData.map(r => r.metric_name)));

            // Build 2D array
            const zValues = [];
            metrics.forEach(metric => {
                const row = [];
                models.forEach(model => {
                    const values = filteredData
                        .filter(r => r.model_id === model && r.metric_name === metric)
                        .map(r => parseFloat(r.value))
                        .filter(v => !isNaN(v));

                    const avg = values.length > 0
                        ? values.reduce((a, b) => a + b, 0) / values.length
                        : null;
                    row.push(avg);
                });
                zValues.push(row);
            });

            const trace = {
                x: models,
                y: metrics,
                z: zValues,
                type: 'heatmap',
                colorscale: [
                    [0, '#dc4e41'],
                    [0.5, '#f7bc38'],
                    [1, '#66cb66']
                ],
                hovertemplate: '<b>Model:</b> %{x}<br><b>Metric:</b> %{y}<br><b>Value:</b> %{z:.4f}<extra></extra>'
            };

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: isDarkMode ? '#c3c7cc' : '#5c6470', size: 12 },
                margin: { t: 20, r: 20, b: 100, l: 150 },
                xaxis: {
                    tickangle: -45,
                    side: 'bottom'
                },
                yaxis: {
                    tickangle: 0
                }
            };

            const config = { responsive: true, displayModeBar: true };

            Plotly.newPlot('heatmap', [trace], layout, config);
        }

        function renderBoxPlot(metric) {
            const container = document.getElementById('dashboard-grid');
            const card = createChartCard(`${metric} - Distribution`, 'boxplot-' + metric);
            container.appendChild(card);

            const metricData = filteredData.filter(r => r.metric_name === metric);

            // Group by model
            const modelGroups = {};
            metricData.forEach(row => {
                if (!modelGroups[row.model_id]) {
                    modelGroups[row.model_id] = [];
                }
                const value = parseFloat(row.value);
                if (!isNaN(value)) {
                    modelGroups[row.model_id].push(value);
                }
            });

            const traces = [];
            Object.keys(modelGroups).forEach((model, idx) => {
                traces.push({
                    y: modelGroups[model],
                    type: 'box',
                    name: model,
                    marker: { color: getChartColor(idx) },
                    boxmean: 'sd'
                });
            });

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: isDarkMode ? '#c3c7cc' : '#5c6470', size: 12 },
                margin: { t: 20, r: 20, b: 80, l: 60 },
                yaxis: {
                    title: metricData[0]?.unit || 'Value',
                    gridcolor: isDarkMode ? '#3a3d42' : '#e5e9f0'
                },
                xaxis: {
                    tickangle: -45
                },
                showlegend: false
            };

            const config = { responsive: true, displayModeBar: true };

            Plotly.newPlot('boxplot-' + metric, traces, layout, config);
        }

        function renderScatterPlot(metric1, metric2) {
            const container = document.getElementById('dashboard-grid');
            const card = createChartCard(`${metric1} vs ${metric2}`, 'scatter-' + metric1 + '-' + metric2);
            container.appendChild(card);

            // Get all runs that have both metrics
            const runGroups = {};
            filteredData.forEach(row => {
                const sourceSuite = row.meta_source_suite || '';
                const key = `${row.model_id}|${row.endpoint}|${sourceSuite}|${row.run_number}`;
                if (!runGroups[key]) {
                    runGroups[key] = {
                        model: row.model_id,
                        run: row.run_number,
                        endpoint: row.endpoint,
                        suite: sourceSuite
                    };
                }
                if (row.metric_name === metric1) {
                    runGroups[key].metric1 = parseFloat(row.value);
                }
                if (row.metric_name === metric2) {
                    runGroups[key].metric2 = parseFloat(row.value);
                }
            });

            // Filter to only runs with both metrics
            const validRuns = Object.values(runGroups).filter(r =>
                r.metric1 !== undefined && r.metric2 !== undefined &&
                !isNaN(r.metric1) && !isNaN(r.metric2)
            );

            // Group by model for coloring
            const modelGroups = {};
            validRuns.forEach(run => {
                if (!modelGroups[run.model]) {
                    modelGroups[run.model] = [];
                }
                modelGroups[run.model].push(run);
            });

            const traces = [];
            Object.keys(modelGroups).forEach((model, idx) => {
                const data = modelGroups[model];
                traces.push({
                    x: data.map(r => r.metric1),
                    y: data.map(r => r.metric2),
                    type: 'scatter',
                    mode: 'markers',
                    name: model,
                    marker: {
                        color: getChartColor(idx),
                        size: 10,
                        opacity: 0.7
                    },
                    hovertemplate: '<b>%{fullData.name}</b><br>' + metric1 + ': %{x:.4f}<br>' + metric2 + ': %{y:.4f}<extra></extra>'
                });
            });

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: isDarkMode ? '#c3c7cc' : '#5c6470', size: 12 },
                margin: { t: 20, r: 20, b: 60, l: 60 },
                xaxis: {
                    title: metric1,
                    gridcolor: isDarkMode ? '#3a3d42' : '#e5e9f0'
                },
                yaxis: {
                    title: metric2,
                    gridcolor: isDarkMode ? '#3a3d42' : '#e5e9f0'
                },
                legend: {
                    orientation: 'h',
                    y: -0.2
                },
                hovermode: 'closest'
            };

            const config = { responsive: true, displayModeBar: true };

            Plotly.newPlot('scatter-' + metric1 + '-' + metric2, traces, layout, config);

            // Add click event to show individual run details
            document.getElementById('scatter-' + metric1 + '-' + metric2).on('plotly_click', (data) => {
                const pointIndex = data.points[0].pointIndex;
                const traceIndex = data.points[0].curveNumber;
                const modelId = Object.keys(modelGroups)[traceIndex];
                const runData = modelGroups[modelId][pointIndex];

                // Find the full run data from metricsData (using full run identity)
                const fullRun = metricsData.find(r => {
                    const rSuite = r.meta_source_suite || '';
                    return r.model_id === runData.model &&
                        r.run_number === runData.run &&
                        r.endpoint === runData.endpoint &&
                        rSuite === runData.suite;
                });

                if (fullRun) {
                    showRunDetails(fullRun);
                }
            });
        }

        function createChartCard(title, chartId) {
            const card = document.createElement('div');
            card.className = 'card';

            // Determine if chart is clickable based on type
            const isClickable = chartId.includes('timeseries') || chartId.includes('scatter');
            const subtitle = isClickable
                ? 'Interactive chart - click points to see run details'
                : 'Interactive chart - hover for information';

            card.innerHTML = `
                <div class="card-header">
                    <div>
                        <div class="card-title">${title}</div>
                        <div class="card-subtitle">${subtitle}</div>
                    </div>
                    <button class="btn" onclick="exportChartPNG('${chartId}')">
                        📷 Export
                    </button>
                </div>
                <div id="${chartId}" class="chart-container"></div>
            `;
            return card;
        }

        function getChartColor(index) {
            const colors = [
                '#00cdaf', '#3aa6dd', '#f7bc38', '#a870ef',
                '#ff6b6b', '#66cb66', '#ff8c00', '#00bcd4',
                '#ff69b4', '#98d8c8'
            ];
            return colors[index % colors.length];
        }

        // ===== MODAL FUNCTIONS =====
        function showRunDetails(run) {
            currentModalRun = run;
            const modal = document.getElementById('modal-overlay');
            const modalTitle = document.getElementById('modal-title');
            const modalBody = document.getElementById('modal-body');

            modalTitle.textContent = `${run.model_id} - Run #${run.run_number}`;

            // Get source suite from metadata
            const sourceSuite = run.meta_source_suite || '';

            // Get all metrics for this UNIQUE run (using full run identity: model + endpoint + suite + run_number)
            const runMetrics = metricsData.filter(r =>
                r.model_id === run.model_id &&
                r.run_number === run.run_number &&
                r.endpoint === run.endpoint &&
                (r.meta_source_suite || '') === sourceSuite
            );

            // Find the row that contains input/output data (usually the row with output_length or any row that has it)
            const runWithData = runMetrics.find(r =>
                r.input_prompt || r.raw_response
            ) || run;

            let html = '';

            // Input Prompt - use runWithData which contains the input/output data
            if (runWithData.input_prompt && runWithData.input_prompt !== '' && runWithData.input_prompt !== '-') {
                const prompt = runWithData.input_prompt.replace(/^"|"$/g, '');
                html += `
                    <div class="modal-section">
                        <div class="modal-section-title">📝 Input Prompt</div>
                        <div class="modal-text">${escapeHtml(prompt)}</div>
                    </div>
                `;
            }

            // Images - handle input and output images (side-by-side for detection/point)
            const hasInputImage = runWithData.input_image_path && runWithData.input_image_path !== 'null' && runWithData.input_image_path !== '' && runWithData.input_image_path !== '-';
            const hasOutputImage = runWithData.output_image_path && runWithData.output_image_path !== 'null' && runWithData.output_image_path !== '' && runWithData.output_image_path !== '-';
            const isDetectionOrPoint = run.endpoint === 'vision/detect' || run.endpoint === 'vision/point';

            if (hasInputImage || hasOutputImage) {
                html += `<div class="modal-section">`;

                // For detection/point: show side-by-side. For others: show input only
                if (isDetectionOrPoint && (hasInputImage || hasOutputImage)) {
                    html += `
                        <div class="modal-section-title">🖼️ ${hasInputImage && hasOutputImage ? 'Input & Detection Results' : hasOutputImage ? 'Detection Results' : 'Input Image'}</div>
                        <div style="display: grid; grid-template-columns: ${hasInputImage && hasOutputImage ? '1fr 1fr' : '1fr'}; gap: 16px; margin-top: 12px;">
                    `;

                    // Input image column
                    if (hasInputImage) {
                        const inputImageId = `modal-input-image-${Date.now()}`;
                        html += `
                            <div>
                                <div style="font-weight: 600; margin-bottom: 8px; color: var(--text-secondary);">Input Image</div>
                                <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px;">Path: ${escapeHtml(runWithData.input_image_path)}</div>
                                <img
                                    id="${inputImageId}"
                                    src="file://${runWithData.input_image_path}"
                                    alt="Input Image"
                                    style="max-width: 100%; height: auto; max-height: 300px; border-radius: 8px; border: 2px solid var(--border-primary); cursor: zoom-in; display: block;"
                                    onclick="expandImage('${inputImageId}')"
                                    onerror="this.style.display='none'; this.nextElementSibling.style.display='block';"
                                />
                                <div style="display: none; padding: 12px; background: var(--bg-tertiary); border-radius: 6px; color: var(--text-muted); font-size: 0.85rem;">
                                    ⚠️ Image preview unavailable.
                                </div>
                            </div>
                        `;
                    }

                    // Output image column (detection/point visualization)
                    if (hasOutputImage) {
                        const outputImageId = `modal-output-image-${Date.now()}`;
                        html += `
                            <div>
                                <div style="font-weight: 600; margin-bottom: 8px; color: var(--accent-primary);">Detections</div>
                                <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px;">Path: ${escapeHtml(runWithData.output_image_path)}</div>
                                <img
                                    id="${outputImageId}"
                                    src="file://${runWithData.output_image_path}"
                                    alt="Detection Output"
                                    style="max-width: 100%; height: auto; max-height: 300px; border-radius: 8px; border: 2px solid var(--accent-primary); cursor: zoom-in; display: block;"
                                    onclick="expandImage('${outputImageId}')"
                                    onerror="this.style.display='none'; this.nextElementSibling.style.display='block';"
                                />
                                <div style="display: none; padding: 12px; background: var(--bg-tertiary); border-radius: 6px; color: var(--text-muted); font-size: 0.85rem;">
                                    ⚠️ Inference failed or no detections found.
                                </div>
                            </div>
                        `;
                    }

                    html += `</div>`;
                } else if (hasInputImage) {
                    // Non-detection/point endpoints: show input image only
                    const imageId = `modal-image-${Date.now()}`;
                    html += `
                        <div class="modal-section-title">🖼️ Input Image</div>
                        <div class="modal-text" style="margin-bottom: 12px;">Path: ${escapeHtml(runWithData.input_image_path)}</div>
                        <div style="text-align: center; margin-top: 12px;">
                            <img
                                id="${imageId}"
                                src="file://${runWithData.input_image_path}"
                                alt="Input Image"
                                style="max-width: 400px; max-height: 300px; border-radius: 8px; border: 2px solid var(--border-primary); cursor: zoom-in; display: block; margin: 0 auto;"
                                onclick="expandImage('${imageId}')"
                                onerror="this.style.display='none'; this.nextElementSibling.style.display='block';"
                            />
                            <div style="display: none; padding: 12px; background: var(--bg-tertiary); border-radius: 6px; color: var(--text-muted); font-size: 0.85rem;">
                                ⚠️ Image preview unavailable.
                            </div>
                        </div>
                    `;
                }

                html += `</div>`;
            }

            // Raw Response - use runWithData
            if (runWithData.raw_response && runWithData.raw_response !== '' && runWithData.raw_response !== '-') {
                const response = runWithData.raw_response.replace(/^"|"$/g, '');
                html += `
                    <div class="modal-section">
                        <div class="modal-section-title">💬 Raw Response</div>
                        <div class="modal-text">${escapeHtml(response)}</div>
                    </div>
                `;
            }

            // Metrics
            html += `
                <div class="modal-section">
                    <div class="modal-section-title">📊 Metrics</div>
                    <div class="modal-metrics">
            `;

            runMetrics.forEach(metric => {
                // Add warmup/counted label if there are multiple entries with same metric name
                const sameMetricCount = runMetrics.filter(m => m.metric_name === metric.metric_name).length;
                const metricLabel = sameMetricCount > 1
                    ? `${metric.metric_name} ${metric.is_warmup === 'True' ? '(Warmup)' : '(Counted)'}`
                    : metric.metric_name;

                html += `
                    <div class="metric-item" title="${metric.is_warmup === 'True' ? 'Warmup iteration - not included in final statistics' : 'Counted run - included in statistics'}">
                        <div class="metric-label">${metricLabel}</div>
                        <div class="metric-value">${parseFloat(metric.value).toFixed(4)} ${metric.unit || ''}</div>
                    </div>
                `;
            });

            html += `
                    </div>
                </div>
            `;

            // Additional Info
            html += `
                <div class="modal-section">
                    <div class="modal-section-title">ℹ️ Additional Info</div>
                    <div class="modal-metrics">
                        <div class="metric-item">
                            <div class="metric-label">Benchmark ID</div>
                            <div class="metric-value" style="font-size: 0.9rem;">${run.benchmark_id || 'N/A'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Suite Type</div>
                            <div class="metric-value" style="font-size: 0.9rem;">${run.suite_type || 'N/A'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Endpoint</div>
                            <div class="metric-value" style="font-size: 0.9rem;">${run.endpoint || 'N/A'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Warmup</div>
                            <div class="metric-value" style="font-size: 0.9rem;">${run.is_warmup === 'True' ? 'Yes' : 'No'}</div>
                        </div>
                    </div>
                </div>
            `;

            modalBody.innerHTML = html;
            modal.classList.add('active');
        }

        function closeModal(event) {
            if (!event || event.target.id === 'modal-overlay') {
                document.getElementById('modal-overlay').classList.remove('active');
                currentModalRun = null;
            }
        }

        function pinCurrentRun() {
            if (!currentModalRun) return;

            const sourceSuite = currentModalRun.meta_source_suite || '';

            // Check if already pinned (using full run identity)
            const existing = pinnedRuns.find(r => {
                const rSuite = r.meta_source_suite || '';
                return r.model_id === currentModalRun.model_id &&
                    r.run_number === currentModalRun.run_number &&
                    r.endpoint === currentModalRun.endpoint &&
                    rSuite === sourceSuite;
            });

            if (existing) {
                showAlert('This run is already pinned!', 'info');
                return;
            }

            pinnedRuns.push(currentModalRun);
            renderPinnedRuns();
            toggleBottomPanel(true);
            showAlert('Run pinned to comparison panel!', 'info');
        }

        function renderPinnedRuns() {
            const container = document.getElementById('panel-content');
            container.innerHTML = '';

            if (pinnedRuns.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-text">No runs pinned yet</div></div>';
                return;
            }

            pinnedRuns.forEach((run, index) => {
                const sourceSuite = run.meta_source_suite || '';

                // Get all metrics for this UNIQUE run
                const runMetrics = metricsData.filter(r =>
                    r.model_id === run.model_id &&
                    r.run_number === run.run_number &&
                    r.endpoint === run.endpoint &&
                    (r.meta_source_suite || '') === sourceSuite
                );

                // Find the row with input/output data
                const runWithData = runMetrics.find(r =>
                    r.input_prompt || r.raw_response
                ) || run;

                const item = document.createElement('div');
                item.className = 'pinned-item';

                let html = `
                    <button class="pinned-item-close" onclick="unpinRun(${index})">&times;</button>
                    <h4 style="color: var(--accent-primary); margin-bottom: 12px;">
                        ${run.model_id} - Run #${run.run_number}
                    </h4>
                `;

                if (runWithData.input_prompt && runWithData.input_prompt !== '' && runWithData.input_prompt !== '-') {
                    const prompt = runWithData.input_prompt.replace(/^"|"$/g, '').substring(0, 150);
                    html += `
                        <div style="margin-bottom: 12px;">
                            <strong style="color: var(--text-secondary); font-size: 0.85rem;">Prompt:</strong>
                            <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;">
                                ${escapeHtml(prompt)}${prompt.length >= 150 ? '...' : ''}
                            </div>
                        </div>
                    `;
                }

                // Add image thumbnail if available
                if (runWithData.input_image_path && runWithData.input_image_path !== 'null' && runWithData.input_image_path !== '' && runWithData.input_image_path !== '-') {
                    const imageId = `pinned-image-${index}`;
                    html += `
                        <div style="margin-bottom: 12px; text-align: center;">
                            <img
                                id="${imageId}"
                                src="file://${runWithData.input_image_path}"
                                alt="Input Image"
                                style="max-width: 150px; max-height: 100px; border-radius: 6px; border: 1px solid var(--border-primary); cursor: zoom-in;"
                                onclick="expandImage('${imageId}')"
                                onerror="this.style.display='none';"
                            />
                        </div>
                    `;
                }

                html += '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px;">';

                runMetrics.forEach(metric => {
                    html += `
                        <div style="background: var(--bg-tertiary); padding: 8px; border-radius: 6px;">
                            <div style="font-size: 0.75rem; color: var(--text-muted);">${metric.metric_name}</div>
                            <div style="font-size: 1rem; font-weight: 700; color: var(--accent-info);">
                                ${parseFloat(metric.value).toFixed(3)} ${metric.unit || ''}
                            </div>
                        </div>
                    `;
                });

                html += '</div>';
                item.innerHTML = html;
                container.appendChild(item);
            });
        }

        function unpinRun(index) {
            pinnedRuns.splice(index, 1);
            renderPinnedRuns();
            if (pinnedRuns.length === 0) {
                toggleBottomPanel(false);
            }
        }

        function clearPinnedRuns() {
            pinnedRuns = [];
            renderPinnedRuns();
            toggleBottomPanel(false);
        }

        function toggleBottomPanel(forceOpen) {
            const panel = document.getElementById('bottom-panel');
            if (forceOpen === true) {
                panel.classList.add('active');
            } else if (forceOpen === false) {
                panel.classList.remove('active');
            } else {
                panel.classList.toggle('active');
            }
        }

        // ===== THEME TOGGLE =====
        function toggleTheme() {
            isDarkMode = !isDarkMode;
            document.body.classList.toggle('light-mode');
            document.getElementById('theme-icon').textContent = isDarkMode ? '🌙' : '☀️';

            // Re-render charts with new theme
            renderCharts();
        }

        // ===== EXPORT FUNCTIONS =====

        function exportChartPNG(chartId) {
            Plotly.downloadImage(chartId, {
                format: 'png',
                width: 1200,
                height: 800,
                filename: chartId
            });
        }

        function exportAllPNG() {
            const chartIds = Object.keys(chartInstances);
            if (chartIds.length === 0) {
                showAlert('No charts to export!', 'warning');
                return;
            }

            chartIds.forEach((id, index) => {
                setTimeout(() => {
                    exportChartPNG(id);
                }, index * 500);
            });

            showAlert(`Exporting ${chartIds.length} charts...`, 'info');
        }

        function downloadFile(filename, content, mimeType) {
            const blob = new Blob([content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        // ===== ALERTS =====
        function showAlert(message, type = 'info') {
            const container = document.getElementById('alerts-container');
            const alert = document.createElement('div');
            alert.className = `alert alert-${type}`;

            const icon = type === 'warning' ? '⚠️' : type === 'info' ? 'ℹ️' : '✅';
            alert.innerHTML = `<span style="font-size: 1.2rem;">${icon}</span> ${message}`;

            container.appendChild(alert);

            setTimeout(() => {
                alert.style.opacity = '0';
                setTimeout(() => alert.remove(), 300);
            }, 5000);
        }

        // ===== UTILITIES =====
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function expandImage(imageId) {
            const img = document.getElementById(imageId);
            if (!img) return;

            // Create fullscreen overlay
            const overlay = document.createElement('div');
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.95);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: zoom-out;
                padding: 20px;
            `;

            // Clone the image for fullscreen display
            const fullImg = img.cloneNode(true);
            fullImg.style.cssText = `
                max-width: 90vw;
                max-height: 90vh;
                width: auto;
                height: auto;
                border-radius: 8px;
                cursor: zoom-out;
                box-shadow: 0 10px 50px rgba(0, 0, 0, 0.5);
            `;
            fullImg.onclick = null; // Remove the expand handler

            overlay.appendChild(fullImg);
            document.body.appendChild(overlay);

            // Close on click
            overlay.addEventListener('click', () => {
                document.body.removeChild(overlay);
            });

            // Close on escape key
            const handleEscape = (e) => {
                if (e.key === 'Escape') {
                    document.body.removeChild(overlay);
                    document.removeEventListener('keydown', handleEscape);
                }
            };
            document.addEventListener('keydown', handleEscape);
        }

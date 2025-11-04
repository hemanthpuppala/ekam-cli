// ===== ENHANCED DASHBOARD FEATURES =====
// This file extends the base dashboard.js with advanced features:
// - Aggregation level filtering
// - Metric deduplication
// - Help sidebar with detailed explanations
// - Recommendations engine
// - Model comparison
// - Interpretation badges
// - Enhanced tooltips

// ===== GLOBAL STATE EXTENSIONS =====
let currentAggregationLevel = 'global';  // 'global', 'per_model', 'per_endpoint'
let comparisonMode = false;
let selectedModelsForComparison = [];
let recommendationsVisible = false;
let currentHelpMetric = null;

// ===== AGGREGATION LEVEL FILTERING =====

function changeAggregationLevel() {
    const select = document.getElementById('aggregation-filter');
    currentAggregationLevel = select.value;

    // Re-render dashboard with new aggregation level
    renderDashboard();

    // Show notification
    showAlert(`Switched to ${select.options[select.selectedIndex].text}`, 'info', 2000);
}

function filterByAggregationLevel(data) {
    // Filter aggregates data based on current aggregation level
    if (!data || data.length === 0) return [];

    return data.filter(row => {
        if (!row.aggregation_level) return true;  // Keep if no aggregation_level field
        return row.aggregation_level === currentAggregationLevel;
    });
}

function deduplicateMetrics(data) {
    // Remove duplicate metrics by keeping only the first occurrence of each metric_name
    // This prevents showing the same metric 3 times (global, per_model, per_endpoint)
    const seen = new Set();
    const deduplicated = [];

    for (const row of data) {
        const key = `${row.metric_name}_${row.model_id || 'ALL'}`;
        if (!seen.has(key)) {
            seen.add(key);
            deduplicated.push(row);
        }
    }

    return deduplicated;
}

// ===== HELP SIDEBAR FUNCTIONS =====

function openHelpSidebar(metricName) {
    const sidebar = document.getElementById('help-sidebar');
    const content = document.getElementById('help-content');
    currentHelpMetric = metricName;

    // Get metric metadata
    const metadata = METRIC_METADATA[metricName];
    if (!metadata) {
        content.innerHTML = `
            <div class="metric-help-section">
                <h4>⚠️ No information available</h4>
                <p>Metadata for metric "${metricName}" is not yet available.</p>
            </div>
        `;
        sidebar.classList.add('active');
        return;
    }

    // Build help content
    let html = `
        <div class="metric-help-section">
            <h4>${getCategoryIcon(metadata.category)} ${metadata.name}</h4>
            <p><strong>Category:</strong> ${CATEGORY_METADATA[metadata.category]?.name || metadata.category}</p>
            <p>${metadata.description}</p>
        </div>

        <div class="metric-help-section">
            <h4>📊 Detailed Explanation</h4>
            <p>${metadata.detailedExplanation}</p>
        </div>

        <div class="metric-help-section">
            <h4>🔢 Calculation</h4>
            <p>${metadata.calculation}</p>
            <div class="help-formula">${metadata.formula}</div>
        </div>
    `;

    // Add interpretation thresholds
    if (metadata.interpretation) {
        html += `
            <div class="metric-help-section">
                <h4>📈 Interpretation Thresholds</h4>
                <div class="help-thresholds">
        `;

        for (const [level, criteria] of Object.entries(metadata.interpretation)) {
            const badge = `<span class="interpretation-badge ${criteria.color}">${criteria.label}</span>`;
            let range = '';
            if (criteria.min !== undefined && criteria.max !== undefined) {
                range = `${criteria.min} - ${criteria.max} ${metadata.unit}`;
            } else if (criteria.min !== undefined) {
                range = `≥ ${criteria.min} ${metadata.unit}`;
            } else if (criteria.max !== undefined) {
                range = `≤ ${criteria.max} ${metadata.unit}`;
            }

            html += `
                <div class="help-threshold-item">
                    ${badge}
                    <div>
                        <div style="font-weight: 600;">${range}</div>
                        <div style="font-size: 0.85rem; color: var(--text-secondary);">${criteria.description}</div>
                    </div>
                </div>
            `;
        }

        html += `
                </div>
            </div>
        `;
    }

    // Add recommendations
    if (metadata.recommendations && Object.keys(metadata.recommendations).length > 0) {
        html += `
            <div class="metric-help-section">
                <h4>💡 Recommendations</h4>
        `;

        for (const [level, recommendations] of Object.entries(metadata.recommendations)) {
            html += `
                <div class="help-recommendations">
                    <h5>When ${level}:</h5>
                    <ul>
                        ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        html += `</div>`;
    }

    // Add related metrics
    if (metadata.relatedMetrics && metadata.relatedMetrics.length > 0) {
        html += `
            <div class="metric-help-section">
                <h4>🔗 Related Metrics</h4>
                <p>Also consider:</p>
                <ul>
                    ${metadata.relatedMetrics.map(m => {
                        const relMeta = METRIC_METADATA[m];
                        return `<li><a href="#" onclick="openHelpSidebar('${m}'); return false;">${relMeta?.name || m}</a></li>`;
                    }).join('')}
                </ul>
            </div>
        `;
    }

    // Add statistical note
    if (metadata.statisticalNote) {
        html += `
            <div class="metric-help-section">
                <h4>📊 Statistical Note</h4>
                <p style="font-style: italic; color: var(--text-muted);">${metadata.statisticalNote}</p>
            </div>
        `;
    }

    // Add limitations if present
    if (metadata.limitations) {
        html += `
            <div class="metric-help-section">
                <h4>⚠️ Limitations</h4>
                <p style="color: var(--accent-warning);">${metadata.limitations}</p>
            </div>
        `;
    }

    content.innerHTML = html;
    sidebar.classList.add('active');

    // Update sidebar title
    document.getElementById('help-title').textContent = metadata.name;
}

function closeHelpSidebar() {
    const sidebar = document.getElementById('help-sidebar');
    sidebar.classList.remove('active');
    currentHelpMetric = null;
}

function getCategoryIcon(category) {
    return CATEGORY_METADATA[category]?.icon || '📊';
}

// ===== INTERPRETATION BADGE GENERATION =====

function createInterpretationBadge(metricName, value) {
    if (!METRIC_METADATA[metricName]) return '';

    const interpretation = getMetricInterpretation(metricName, value);
    return `<span class="interpretation-badge ${interpretation.color}">${interpretation.label}</span>`;
}

// ===== RECOMMENDATIONS ENGINE =====

function toggleRecommendations() {
    recommendationsVisible = !recommendationsVisible;
    const panel = document.getElementById('recommendations-panel');
    const btn = document.getElementById('recommendations-btn');

    if (recommendationsVisible) {
        generateRecommendations();
        panel.classList.add('active');
        btn.style.background = 'var(--accent-warning)';
        btn.style.color = 'var(--bg-primary)';
    } else {
        panel.classList.remove('active');
        btn.style.background = '';
        btn.style.color = '';
    }
}

function generateRecommendations() {
    const content = document.getElementById('recommendations-content');

    // Analyze aggregates data for recommendations
    const recommendations = [];

    for (const row of aggregatesData) {
        if (row.aggregation_type !== 'counted') continue;  // Only analyze non-warmup runs

        const metricName = row.metric_name;
        const value = parseFloat(row.mean);

        if (!METRIC_METADATA[metricName] || isNaN(value)) continue;

        const recs = getMetricRecommendations(metricName, value);
        if (recs && recs.length > 0) {
            const interpretation = getMetricInterpretation(metricName, value);
            const metadata = METRIC_METADATA[metricName];

            recommendations.push({
                metric: metricName,
                metricDisplayName: metadata.name,
                category: metadata.category,
                value: value,
                unit: metadata.unit,
                interpretation: interpretation,
                recommendations: recs,
                modelId: row.model_id
            });
        }
    }

    // Sort by severity (poor > fair > good)
    const severityOrder = { poor: 0, fair: 1, good: 2, excellent: 3 };
    recommendations.sort((a, b) => {
        const severityA = severityOrder[a.interpretation.label.toLowerCase()] || 99;
        const severityB = severityOrder[b.interpretation.label.toLowerCase()] || 99;
        return severityA - severityB;
    });

    // Render recommendations
    if (recommendations.length === 0) {
        content.innerHTML = `
            <div style="text-align: center; padding: 3rem; color: var(--text-secondary);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
                <h3>All metrics look good!</h3>
                <p>No specific recommendations at this time.</p>
            </div>
        `;
        return;
    }

    let html = '';
    for (const rec of recommendations) {
        const severityClass = rec.interpretation.label.toLowerCase() === 'poor' ? 'high' :
                             rec.interpretation.label.toLowerCase() === 'fair' ? 'medium' : 'low';

        html += `
            <div class="recommendation-item">
                <h4>
                    ${getCategoryIcon(rec.category)} ${rec.metricDisplayName}
                    <span class="recommendation-severity ${severityClass}">${rec.interpretation.label}</span>
                </h4>
                <p style="color: var(--text-secondary); margin-bottom: 0.5rem;">
                    Current value: <strong>${formatMetricValue(rec.value, rec.unit)}</strong>
                    ${rec.modelId && rec.modelId !== 'ALL' ? ` (Model: ${rec.modelId.split('/').pop()})` : ''}
                </p>
                <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.75rem;">
                    ${rec.interpretation.description}
                </p>
                <ul>
                    ${rec.recommendations.map(r => `<li>${r}</li>`).join('')}
                </ul>
                <button class="info-button" onclick="openHelpSidebar('${rec.metric}')">
                    ℹ️ Learn More
                </button>
            </div>
        `;
    }

    content.innerHTML = html;
}

// ===== MODEL COMPARISON MODE =====

function toggleComparisonMode() {
    comparisonMode = !comparisonMode;
    const panel = document.getElementById('comparison-panel');
    const btn = document.getElementById('comparison-btn');

    if (comparisonMode) {
        initializeComparison();
        panel.classList.add('active');
        btn.style.background = 'var(--accent-primary)';
        btn.style.color = 'var(--bg-primary)';
    } else {
        panel.classList.remove('active');
        btn.style.background = '';
        btn.style.color = '';
    }
}

function initializeComparison() {
    const selectorsDiv = document.getElementById('comparison-model-selectors');
    const modelsArray = Array.from(allModels);

    // Create checkboxes for each model
    let html = '';
    for (const model of modelsArray) {
        const shortName = model.split('/').pop();
        const checked = selectedModelsForComparison.includes(model) ? 'checked' : '';
        html += `
            <label class="model-checkbox">
                <input type="checkbox" value="${model}" ${checked} onchange="updateComparisonSelection()">
                <span>${shortName}</span>
            </label>
        `;
    }

    selectorsDiv.innerHTML = html;

    // Render comparison if models are selected
    if (selectedModelsForComparison.length >= 2) {
        renderComparison();
    } else {
        document.getElementById('comparison-content').innerHTML = `
            <div style="text-align: center; padding: 3rem; color: var(--text-secondary);">
                <p>Select at least 2 models to compare</p>
            </div>
        `;
    }
}

function updateComparisonSelection() {
    const checkboxes = document.querySelectorAll('#comparison-model-selectors input[type="checkbox"]');
    selectedModelsForComparison = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    if (selectedModelsForComparison.length >= 2) {
        renderComparison();
    } else {
        document.getElementById('comparison-content').innerHTML = `
            <div style="text-align: center; padding: 3rem; color: var(--text-secondary);">
                <p>Select at least 2 models to compare</p>
            </div>
        `;
    }
}

function renderComparison() {
    const content = document.getElementById('comparison-content');

    // Get metrics for selected models
    const comparisonData = {};

    for (const row of aggregatesData) {
        if (row.aggregation_level !== 'per_model') continue;
        if (!selectedModelsForComparison.includes(row.model_id)) continue;
        if (row.aggregation_type !== 'counted') continue;  // Only non-warmup

        const metricName = row.metric_name;
        if (!comparisonData[metricName]) {
            comparisonData[metricName] = {};
        }

        comparisonData[metricName][row.model_id] = {
            mean: parseFloat(row.mean),
            median: parseFloat(row.median),
            std_dev: parseFloat(row.std_dev)
        };
    }

    // Build comparison table
    let html = '<div style="padding: 1.5rem;">';

    for (const [metricName, modelData] of Object.entries(comparisonData)) {
        if (Object.keys(modelData).length < 2) continue;  // Skip if not all models have this metric

        const metadata = METRIC_METADATA[metricName];
        if (!metadata) continue;

        // Determine winner (lower is better for latency, memory; higher is better for consistency)
        const lowerIsBetter = ['latency', 'memory', 'cpu', 'degradation'].some(term => metricName.includes(term));

        const values = Object.entries(modelData).map(([model, data]) => ({ model, value: data.mean }));
        values.sort((a, b) => lowerIsBetter ? a.value - b.value : b.value - a.value);
        const winnerModel = values[0].model;

        html += `
            <div style="margin-bottom: 2rem;">
                <h4 style="color: var(--text-primary); margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between;">
                    ${metadata.name}
                    <button class="info-button" onclick="openHelpSidebar('${metricName}')">ℹ️ Info</button>
                </h4>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Mean</th>
                            <th>Median</th>
                            <th>Std Dev</th>
                            <th>Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        for (const [modelId, data] of Object.entries(modelData)) {
            const shortName = modelId.split('/').pop();
            const isWinner = modelId === winnerModel;
            const rowClass = isWinner ? 'comparison-winner' : 'comparison-loser';
            const badge = createInterpretationBadge(metricName, data.mean);

            html += `
                <tr class="${rowClass}">
                    <td>${shortName} ${isWinner ? '🏆' : ''}</td>
                    <td>${formatMetricValue(data.mean, metadata.unit)}</td>
                    <td>${formatMetricValue(data.median, metadata.unit)}</td>
                    <td>${formatMetricValue(data.std_dev, metadata.unit)}</td>
                    <td>${badge}</td>
                </tr>
            `;
        }

        html += `
                    </tbody>
                </table>
            </div>
        `;
    }

    html += '</div>';
    content.innerHTML = html;
}

// ===== ENHANCED TOOLTIPS =====

// Override the existing tooltip function to use metadata
const originalShowTooltip = typeof showCustomTooltip !== 'undefined' ? showCustomTooltip : null;

function showEnhancedTooltip(event, metricName, value) {
    const tooltip = document.getElementById('custom-tooltip');
    const metadata = METRIC_METADATA[metricName];

    if (!metadata) {
        // Fallback to original tooltip if available
        if (originalShowTooltip) {
            originalShowTooltip(event, metricName, value);
        }
        return;
    }

    const interpretation = getMetricInterpretation(metricName, value);
    const formattedValue = formatMetricValue(value, metadata.unit);

    tooltip.innerHTML = `
        <div class="enhanced-tooltip">
            <div class="tooltip-title">${getCategoryIcon(metadata.category)} ${metadata.name}</div>
            <div class="tooltip-description">${metadata.description}</div>
            <div class="tooltip-value">
                <div class="tooltip-value-number">${formattedValue}</div>
                ${createInterpretationBadge(metricName, value)}
            </div>
        </div>
    `;

    tooltip.style.display = 'block';
    tooltip.style.left = event.pageX + 10 + 'px';
    tooltip.style.top = event.pageY + 10 + 'px';
}

function hideEnhancedTooltip() {
    const tooltip = document.getElementById('custom-tooltip');
    tooltip.style.display = 'none';
}

// ===== EXPORT DETAILED REPORT =====

function exportDetailedReport() {
    // Generate a comprehensive HTML report with MODEL COMPARISON and winners
    let html = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VLM Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #00cdaf; border-bottom: 3px solid #00cdaf; padding-bottom: 0.5rem; }
        h2 { color: #333; margin-top: 2rem; border-bottom: 2px solid #ddd; padding-bottom: 0.5rem; }
        h3 { color: #555; margin-top: 1.5rem; }
        .metric-section { margin: 2rem 0; padding: 1.5rem; border-left: 4px solid #3aa6dd; background: #f9f9f9; }
        .badge { padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold; text-transform: uppercase; font-size: 0.85rem; display: inline-block; }
        .excellent { background: #d4edda; color: #155724; }
        .good { background: #d1ecf1; color: #0c5460; }
        .fair { background: #fff3cd; color: #856404; }
        .poor { background: #f8d7da; color: #721c24; }
        .unknown { background: #e2e3e5; color: #383d41; }
        table { width: 100%; border-collapse: collapse; margin: 1rem 0; background: white; }
        th, td { padding: 0.75rem; text-align: left; border: 1px solid #ddd; }
        th { background: #f0f0f0; font-weight: bold; }
        .winner { background: #e8f8f5; font-weight: bold; }
        .winner-cell { position: relative; }
        .winner-icon { font-size: 1.2rem; margin-right: 0.5rem; }
        .formula { background: #f5f5f5; padding: 1rem; font-family: monospace; border-left: 3px solid #00cdaf; margin: 1rem 0; }
        .model-id { font-family: monospace; font-size: 0.9rem; color: #666; }
        .summary-box { background: #e7f3ff; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #00cdaf; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 VLM Benchmark Detailed Report</h1>
        <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>

        <h2>📋 Executive Summary</h2>
        <p>This report provides a comprehensive model comparison with detailed explanations of each metric, showing which model performs best for each benchmark.</p>
    `;

    // Group metrics by category, collecting ALL models for each metric
    const metricsByCategory = {};

    // First, collect all per-model data
    for (const row of aggregatesData.filter(r => r.aggregation_level === 'per_model' && r.aggregation_type === 'counted')) {
        const metricName = row.metric_name;
        const metadata = METRIC_METADATA[metricName];
        if (!metadata) continue;

        if (!metricsByCategory[metadata.category]) {
            metricsByCategory[metadata.category] = {};
        }

        if (!metricsByCategory[metadata.category][metricName]) {
            metricsByCategory[metadata.category][metricName] = {
                metadata: metadata,
                models: []
            };
        }

        metricsByCategory[metadata.category][metricName].models.push({
            model_id: row.model_id || 'Unknown',
            mean: parseFloat(row.mean),
            median: parseFloat(row.median),
            std_dev: parseFloat(row.std_dev),
            interpretation: getMetricInterpretation(metricName, parseFloat(row.mean))
        });
    }

    // Generate report sections by category
    for (const [category, metrics] of Object.entries(metricsByCategory)) {
        const categoryMeta = CATEGORY_METADATA[category];
        html += `
            <h2>${categoryMeta.icon} ${categoryMeta.name}</h2>
        `;

        for (const [metricName, metricData] of Object.entries(metrics)) {
            const metadata = metricData.metadata;

            // Determine if lower values are better for this metric (used for tiebreakers and worst performer)
            const lowerIsBetter = metricName.includes('latency') || metricName.includes('degradation') ||
                                 metricName.includes('cpu') || metricName.includes('memory');

            // Determine winner based on INTERPRETATION QUALITY (not raw values!)
            // Ranking: excellent > good > fair > poor > unknown
            const interpretationRank = {
                'excellent': 5,
                'good': 4,
                'fair': 3,
                'poor': 2,
                'unknown': 1
            };

            let winnerModel = null;
            if (metricData.models.length > 0) {
                winnerModel = metricData.models.reduce((best, current) => {
                    if (!best) return current;

                    const bestRank = interpretationRank[best.interpretation.label.toLowerCase()] || 0;
                    const currentRank = interpretationRank[current.interpretation.label.toLowerCase()] || 0;

                    // Higher interpretation rank wins
                    if (currentRank > bestRank) {
                        return current;
                    } else if (currentRank < bestRank) {
                        return best;
                    }

                    // If tied on interpretation, use raw value as tiebreaker
                    if (lowerIsBetter) {
                        return current.mean < best.mean ? current : best;
                    } else {
                        // For metrics with optimal ranges (like output_length), pick closest to middle of "excellent" range
                        return current.mean < best.mean ? current : best;
                    }
                }, null);
            }

            html += `
                <div class="metric-section">
                    <h3>${metadata.name}</h3>
                    <p>${metadata.description}</p>

                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Mean</th>
                                <th>Median</th>
                                <th>Std Dev</th>
                                <th>Interpretation</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            // Add rows for each model
            for (const model of metricData.models) {
                const isWinner = winnerModel && model.model_id === winnerModel.model_id;
                const rowClass = isWinner ? 'winner' : '';
                const winnerIcon = isWinner ? '<span class="winner-icon">🏆</span>' : '';
                const modelName = model.model_id.split('/').pop() || model.model_id;

                html += `
                    <tr class="${rowClass}">
                        <td class="winner-cell">${winnerIcon}<span class="model-id">${modelName}</span></td>
                        <td>${formatMetricValue(model.mean, metadata.unit)}</td>
                        <td>${formatMetricValue(model.median, metadata.unit)}</td>
                        <td>${formatMetricValue(model.std_dev, metadata.unit)}</td>
                        <td><span class="badge ${model.interpretation.color}">${model.interpretation.label}</span></td>
                    </tr>
                `;
            }

            html += `
                        </tbody>
                    </table>

                    <p><strong>How it's calculated:</strong> ${metadata.calculation}</p>
                    <div class="formula">${metadata.formula}</div>
            `;

            // Add interpretation guide
            if (metadata.interpretation) {
                html += `<p><strong>Interpretation Guide:</strong></p><ul>`;
                for (const [level, criteria] of Object.entries(metadata.interpretation)) {
                    const rangeText = criteria.min !== undefined && criteria.max !== undefined
                        ? `${criteria.min} - ${criteria.max} ${metadata.unit}`
                        : criteria.min !== undefined
                            ? `≥ ${criteria.min} ${metadata.unit}`
                            : criteria.max !== undefined
                                ? `≤ ${criteria.max} ${metadata.unit}`
                                : 'N/A';
                    html += `<li><span class="badge ${criteria.color}">${criteria.label}</span>: ${rangeText} - ${criteria.description}</li>`;
                }
                html += `</ul>`;
            }

            // Add recommendations for the worst performer
            if (winnerModel) {
                const worstModel = metricData.models.reduce((worst, current) => {
                    if (!worst) return current;
                    if (lowerIsBetter) {
                        return current.mean > worst.mean ? current : worst;
                    } else {
                        return current.mean < worst.mean ? current : worst;
                    }
                }, null);

                const recommendations = getMetricRecommendations(metricName, worstModel.mean);
                if (recommendations && recommendations.length > 0) {
                    const worstModelName = worstModel.model_id.split('/').pop() || worstModel.model_id;
                    html += `
                        <div class="summary-box">
                            <p><strong>Recommendations for ${worstModelName}:</strong></p>
                            <ul>`;
                    for (const rec of recommendations) {
                        html += `<li>${rec}</li>`;
                    }
                    html += `</ul></div>`;
                }
            }

            html += `</div>`;
        }
    }

    html += `
    </div>
</body>
</html>
    `;

    // Download as HTML file
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark-report-${new Date().toISOString().slice(0,10)}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showAlert('Report exported successfully with model comparisons!', 'success', 3000);
}

// ===== INITIALIZATION =====

// Override the existing renderDashboard function to use enhanced features
if (typeof renderDashboard !== 'undefined') {
    const originalRenderDashboard = renderDashboard;

    renderDashboard = function() {
        // Filter aggregates by current aggregation level before rendering
        const originalAggregates = aggregatesData;
        aggregatesData = filterByAggregationLevel(originalAggregates);

        // Call original render
        originalRenderDashboard();

        // Restore original data
        aggregatesData = originalAggregates;

        // Enhance rendered metrics with badges and info buttons
        enhanceRenderedMetrics();
    };
}

function enhanceRenderedMetrics() {
    // This function adds interpretation badges and info buttons to rendered metrics
    // It runs after the base dashboard rendering is complete

    // Find all metric cards and enhance them
    const statsCards = document.querySelectorAll('.stat-card, .metric-card');

    for (const card of statsCards) {
        const metricNameElement = card.querySelector('h3, .metric-name');
        if (!metricNameElement) continue;

        const metricText = metricNameElement.textContent.trim();

        // Try to find matching metric
        let metricName = null;
        for (const key of Object.keys(METRIC_METADATA)) {
            if (METRIC_METADATA[key].name === metricText) {
                metricName = key;
                break;
            }
        }

        if (!metricName) continue;

        // Get the value
        const valueElement = card.querySelector('.stat-value, .metric-value');
        if (!valueElement) continue;

        const valueText = valueElement.textContent.trim();
        const value = parseFloat(valueText.replace(/[^0-9.-]/g, ''));

        if (isNaN(value)) continue;

        // Add interpretation badge
        const badge = createInterpretationBadge(metricName, value);
        if (badge && !card.querySelector('.interpretation-badge')) {
            const badgeContainer = document.createElement('div');
            badgeContainer.innerHTML = badge;
            badgeContainer.style.marginTop = '0.5rem';
            valueElement.after(badgeContainer.firstChild);
        }

        // Add info button
        if (!card.querySelector('.info-button')) {
            const infoBtn = document.createElement('button');
            infoBtn.className = 'info-button';
            infoBtn.innerHTML = 'ℹ️ Info';
            infoBtn.onclick = () => openHelpSidebar(metricName);
            card.appendChild(infoBtn);
        }
    }
}

console.log('✨ Enhanced dashboard features loaded');

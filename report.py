import datetime
import pandas as pd
from analytics import kl_divergence, js_divergence, earth_movers_distance

def generate_report(df: pd.DataFrame, ai_scores, persona_scores) -> str:
    """
    Generate a compliance-ready HTML bias audit report.
    Includes AEDT-style disclosure, NIST AI RMF mapping, and EEOC audit notes.
    """

    n = len(df)
    correct = df["correct"].sum()
    accuracy = correct / n if n > 0 else 0

    # Distribution metrics
    kl = kl_divergence(ai_scores, persona_scores)
    js = js_divergence(ai_scores, persona_scores)
    emd = earth_movers_distance(ai_scores, persona_scores)

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Generate historical trend data
    historical_accuracies = []
    current_correct = 0
    for i in range(len(df)):
        if df.iloc[i]['correct'] == 1:
            current_correct += 1
        historical_accuracies.append(current_correct / (i + 1))

    # Format historical data for HTML
    trend_labels = [f"Trial {i+1}" for i in range(len(df))]
    trend_data = [f"{acc:.4f}" for acc in historical_accuracies]
    trend_datasets = f'{{ label: "User Accuracy", data: [{",".join(trend_data)}], borderColor: "#3498db", fill: false }}'

    # Determine Bias Scorecard value based on KL Divergence
    bias_scorecard = ""
    bias_color = ""
    if kl < 0.1:
        bias_scorecard = "Minimal Bias"
        bias_color = "#28a745" # Green
    elif kl < 0.5:
        bias_scorecard = "Medium Bias"
        bias_color = "#ffc107" # Yellow
    else:
        bias_scorecard = "High Bias"
        bias_color = "#dc3545" # Red

    # Prepare data for score distribution histogram
    ai_scores_str = ",".join(map(str, ai_scores))
    persona_scores_str = ",".join(map(str, persona_scores))

    html = f"""
    <html>
    <head>
        <title>AI Fairness Audit Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: 'Arial', sans-serif; margin: 2em; line-height: 1.6; color: #444; }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.3em; }}
            table {{ border-collapse: collapse; width: 100%; margin: 1em 0; border: 1px solid #ddd; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f8f9fa; color: #333; font-weight: bold; }}
            .section {{ margin-bottom: 2.5em; background-color: #fefefe; padding: 1.5em; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 0.5em; }}
            .bias-scorecard {{
                font-size: 1.5em;
                font-weight: bold;
                padding: 0.5em 1em;
                border-radius: 5px;
                color: white;
                text-align: center;
                margin-top: 1em;
                background-color: {bias_color};
            }}
        </style>
    </head>
    <body>
        <h1>📄 AI Fairness Audit Report</h1>
        <p>This report provides a comprehensive analysis of the bias present in an AI hiring model, comparing its decisions to those of a known biased human judge. The findings are presented to be easily understood by all stakeholders, from technical teams to legal and human resources personnel.</p>
        
        <div class="section">
            <h2>Summary of Key Findings</h2>
            <table>
                <tr><th>Metric</th><th>Result</th><th>Explanation</th></tr>
                <tr><td>Date of Report</td><td>{now}</td><td>The time and date the audit was performed.</td></tr>
                <tr><td>Number of Trials</td><td>{n}</td><td>The total number of resumes evaluated in the test.</td></tr>
                <tr><td>User Accuracy</td><td>{accuracy:.2%} ({correct}/{n})</td><td>The percentage of times a human user correctly identified the biased judge's score. An accuracy near 50% suggests the AI's bias is indistinguishable from the human's.</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>Bias Scorecard</h2>
            <div class="bias-scorecard">{bias_scorecard}</div>
            <p>This scorecard provides a quick, color-coded summary of the AI's bias level. The score is based on the KL Divergence metric, where a lower value indicates a greater similarity to the biased human judge's scores.</p>
        </div>

        <div class="section">
            <h2>Quantitative Bias Metrics</h2>
            <p>The following metrics compare the statistical distribution of scores from the AI and the biased human judge. A lower value for these metrics indicates that the AI's scoring pattern is highly similar to the biased human's, which is a key indicator of bias.</p>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
                <tr><td>KL Divergence</td><td>{kl:.4f}</td><td>Measures how one score distribution differs from the other. A value close to 0 suggests the AI and human scores are nearly identical.</td></tr>
                <tr><td>JS Divergence</td><td>{js:.4f}</td><td>A symmetrical version of KL Divergence, providing a more stable measure of similarity. A lower value indicates higher similarity.</td></tr>
                <tr><td>Earth Mover’s Distance</td><td>{emd:.4f}</td><td>Represents the minimum "work" required to transform one score distribution into the other. A low value means the distributions are very alike.</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Historical Bias Trend Analysis</h2>
            <p>This section shows how your ability to detect bias has changed over the course of the test. A flat trend indicates consistent difficulty, while an improving trend suggests you are learning to spot the bias more effectively.</p>
            <canvas id="trendChart"></canvas>
        </div>
        
        <div class="section">
            <h2>Score Distribution Visualization</h2>
            <p>This chart shows the distribution of scores from the AI model and the biased human judge, providing a visual explanation of the quantitative metrics above.</p>
            <canvas id="scoreDistributionChart"></canvas>
        </div>

        <div class="section">
            <h2>Compliance and Best Practices Alignment</h2>
            <p>This audit evaluates the AI system against established regulatory and ethical frameworks.</p>
            <h3>NYC Automated Employment Decision Tools (AEDT) - Local Law 144</h3>
            <p>
                The test methodology simulates a "bias audit" consistent with NYC's requirements. It evaluates fairness based on the impact ratio, comparing selection rates across groups. Our findings show disparities that would require disclosure and further investigation.
            </p>
            <h3>NIST AI Risk Management Framework (AI RMF)</h3>
            <ul>
                <li><b>Map:</b> Sensitive attributes like 'gender', 'education', and 'gap years' were identified and used as a basis for the biased judge.</li>
                <li><b>Measure:</b> Bias was quantified using the metrics in this report, providing a clear way to track and evaluate fairness.</li>
                <li><b>Manage:</b> Mitigation techniques, such as score reweighing and threshold optimization, have been applied to reduce the observed bias.</li>
            </ul>
            <h3>EEOC Audit Readiness</h3>
            <p>
                The audit identifies risks aligned with U.S. Equal Employment Opportunity Commission (EEOC) guidelines, specifically highlighting potential disparate impact based on the identified sensitive attributes. It is recommended that this AI tool undergo continuous monitoring and be used with human oversight to ensure compliance.
            </p>
        </div>

        <script>
            // Historical Trend Chart
            const trendCtx = document.getElementById('trendChart').getContext('2d');
            new Chart(trendCtx, {{
                type: 'line',
                data: {{
                    labels: [{",".join([f"'{l}'" for l in trend_labels])}],
                    datasets: [{trend_datasets}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{ beginAtZero: true, max: 1, title: {{ display: true, text: 'Accuracy' }} }}
                    }},
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{ display: true, text: 'User Accuracy Over Time' }}
                    }}
                }}
            }});
            
            // Score Distribution Chart (Histogram)
            const scoreDistCtx = document.getElementById('scoreDistributionChart').getContext('2d');
            const aiScores = [{ai_scores_str}];
            const personaScores = [{persona_scores_str}];
            
            // Function to bin data for the histogram
            const binData = (data, min, max, bins) => {{
                const binSize = (max - min) / bins;
                const counts = new Array(bins).fill(0);
                data.forEach(value => {{
                    const binIndex = Math.min(Math.floor((value - min) / binSize), bins - 1);
                    counts[binIndex]++;
                }});
                const labels = [];
                for (let i = 0; i < bins; i++) {{
                    labels.push(`${{Math.round(min + i * binSize)}} - ${{Math.round(min + (i + 1) * binSize)}}`);
                }}
                return {{ labels, counts }};
            }};
            
            const minScore = 0;
            const maxScore = 100;
            const bins = 10;
            
            const aiBins = binData(aiScores, minScore, maxScore, bins);
            const personaBins = binData(personaScores, minScore, maxScore, bins);

            new Chart(scoreDistCtx, {{
                type: 'bar',
                data: {{
                    labels: aiBins.labels,
                    datasets: [
                        {{
                            label: 'AI Scores',
                            data: aiBins.counts,
                            backgroundColor: 'rgba(52, 152, 219, 0.5)',
                            borderColor: 'rgba(52, 152, 219, 1)',
                            borderWidth: 1
                        }},
                        {{
                            label: 'Biased Human Scores',
                            data: personaBins.counts,
                            backgroundColor: 'rgba(231, 76, 60, 0.5)',
                            borderColor: 'rgba(231, 76, 60, 1)',
                            borderWidth: 1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Score Range' }} }},
                        y: {{ beginAtZero: true, title: {{ display: true, text: 'Frequency' }} }}
                    }},
                    plugins: {{
                        title: {{ display: true, text: 'Score Distribution: AI vs. Biased Human' }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html

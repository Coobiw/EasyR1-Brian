#!/usr/bin/env python3
"""
HTML-based visualization script for comparing different reward functions.
This version generates an interactive HTML file that can be opened in any browser.
No matplotlib or complex dependencies required!
"""

import sys
import os
import math

# Add examples/score_function to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples', 'score_function'))

# Import reward functions from agiqa3k module
from agiqa3k import grade_answer_l1, grade_answer_l2, grade_answer_laplace, grade_answer_gaussian


def generate_plot_data():
    """Generate data points for all reward functions."""
    
    # Set up the ground truth value
    gt = 3.0
    
    # Generate prediction values from 1.0 to 5.0
    num_points = 200
    pred_values = [1.0 + i * (5.0 - 1.0) / (num_points - 1) for i in range(num_points)]
    
    # Parameters for all functions
    r_min = 0.05
    diff_at_rmin = 1.0
    use_floor = True
    
    # Calculate rewards for each function
    data = {
        'pred_values': pred_values,
        'l1': [grade_answer_l1(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values],
        'l2': [grade_answer_l2(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values],
        'laplace': [grade_answer_laplace(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values],
        'gaussian': [grade_answer_gaussian(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values],
        'gt': gt,
        'r_min': r_min,
        'diff_at_rmin': diff_at_rmin,
    }
    
    return data


def create_html_visualization():
    """Create an interactive HTML visualization using Plotly.js."""
    
    print("Generating data...")
    data = generate_plot_data()
    
    # Convert data to JSON-like format for JavaScript
    pred_str = str(data['pred_values'])
    l1_str = str(data['l1'])
    l2_str = str(data['l2'])
    laplace_str = str(data['laplace'])
    gaussian_str = str(data['gaussian'])
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Reward Functions Comparison</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .plot-container {{
            margin: 20px 0;
        }}
        .info {{
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .info h3 {{
            margin-top: 0;
            color: #1976d2;
        }}
        .info ul {{
            margin: 10px 0;
        }}
        .description {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .func-card {{
            border: 2px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #fafafa;
        }}
        .func-card h4 {{
            margin-top: 0;
            border-bottom: 2px solid;
            padding-bottom: 5px;
        }}
        .func-card.l1 h4 {{ border-color: #1f77b4; color: #1f77b4; }}
        .func-card.l2 h4 {{ border-color: #ff7f0e; color: #ff7f0e; }}
        .func-card.laplace h4 {{ border-color: #2ca02c; color: #2ca02c; }}
        .func-card.gaussian h4 {{ border-color: #d62728; color: #d62728; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Reward Functions Comparison</h1>
        
        <div class="info">
            <h3>Configuration</h3>
            <ul>
                <li><strong>Ground Truth (GT):</strong> {data['gt']}</li>
                <li><strong>Minimum Reward (r_min):</strong> {data['r_min']}</li>
                <li><strong>Diff at r_min:</strong> {data['diff_at_rmin']}</li>
                <li><strong>Prediction Range:</strong> [1.0, 5.0]</li>
            </ul>
        </div>

        <div class="plot-container">
            <div id="overlayPlot"></div>
        </div>

        <div class="plot-container">
            <div id="subplots"></div>
        </div>

        <div class="description">
            <div class="func-card l1">
                <h4>L1 (Linear)</h4>
                <p><strong>Formula:</strong> reward = 1 - coeff × |pred - gt|</p>
                <p><strong>Characteristic:</strong> Linear decay with uniform penalty for errors.</p>
            </div>
            
            <div class="func-card l2">
                <h4>L2 (Quadratic)</h4>
                <p><strong>Formula:</strong> reward = 1 - coeff × (pred - gt)²</p>
                <p><strong>Characteristic:</strong> Quadratic decay, gentle for small errors, severe for large ones.</p>
            </div>
            
            <div class="func-card laplace">
                <h4>Laplace</h4>
                <p><strong>Formula:</strong> reward = exp(-d / τ)</p>
                <p><strong>Characteristic:</strong> Exponential decay with heavier tails than Gaussian.</p>
            </div>
            
            <div class="func-card gaussian">
                <h4>Gaussian</h4>
                <p><strong>Formula:</strong> reward = exp(-(d² / (2σ²)))</p>
                <p><strong>Characteristic:</strong> Bell-shaped curve with smooth decay.</p>
            </div>
        </div>
    </div>

    <script>
        // Data
        const predValues = {pred_str};
        const l1Values = {l1_str};
        const l2Values = {l2_str};
        const laplaceValues = {laplace_str};
        const gaussianValues = {gaussian_str};
        const gt = {data['gt']};
        const rMin = {data['r_min']};

        // Overlay Plot
        const overlayTraces = [
            {{
                x: predValues,
                y: l1Values,
                mode: 'lines',
                name: 'L1 (Linear)',
                line: {{ color: '#1f77b4', width: 2.5 }}
            }},
            {{
                x: predValues,
                y: l2Values,
                mode: 'lines',
                name: 'L2 (Quadratic)',
                line: {{ color: '#ff7f0e', width: 2.5 }}
            }},
            {{
                x: predValues,
                y: laplaceValues,
                mode: 'lines',
                name: 'Laplace',
                line: {{ color: '#2ca02c', width: 2.5 }}
            }},
            {{
                x: predValues,
                y: gaussianValues,
                mode: 'lines',
                name: 'Gaussian',
                line: {{ color: '#d62728', width: 2.5 }}
            }}
        ];

        const overlayLayout = {{
            title: 'Overlay Comparison of Reward Functions',
            xaxis: {{ 
                title: 'Prediction Value',
                range: [1.0, 5.0],
                gridcolor: '#e0e0e0'
            }},
            yaxis: {{ 
                title: 'Reward',
                range: [0, 1.05],
                gridcolor: '#e0e0e0'
            }},
            shapes: [
                {{
                    type: 'line',
                    x0: gt,
                    y0: 0,
                    x1: gt,
                    y1: 1.05,
                    line: {{
                        color: 'gray',
                        width: 2,
                        dash: 'dash'
                    }}
                }},
                {{
                    type: 'line',
                    x0: 1.0,
                    y0: rMin,
                    x1: 5.0,
                    y1: rMin,
                    line: {{
                        color: 'black',
                        width: 1,
                        dash: 'dot'
                    }}
                }}
            ],
            annotations: [
                {{
                    x: gt,
                    y: 1.02,
                    text: 'GT=' + gt,
                    showarrow: false,
                    font: {{ color: 'gray' }}
                }},
                {{
                    x: 4.8,
                    y: rMin + 0.05,
                    text: 'r_min=' + rMin,
                    showarrow: false,
                    font: {{ color: 'black', size: 10 }}
                }}
            ],
            hovermode: 'x unified',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: {{ family: 'Arial' }}
        }};

        Plotly.newPlot('overlayPlot', overlayTraces, overlayLayout, {{responsive: true}});

        // Subplots
        const subplotTraces = [
            {{
                x: predValues,
                y: l1Values,
                mode: 'lines',
                name: 'L1',
                line: {{ color: '#1f77b4', width: 2.5 }},
                xaxis: 'x1',
                yaxis: 'y1'
            }},
            {{
                x: predValues,
                y: l2Values,
                mode: 'lines',
                name: 'L2',
                line: {{ color: '#ff7f0e', width: 2.5 }},
                xaxis: 'x2',
                yaxis: 'y2'
            }},
            {{
                x: predValues,
                y: laplaceValues,
                mode: 'lines',
                name: 'Laplace',
                line: {{ color: '#2ca02c', width: 2.5 }},
                xaxis: 'x3',
                yaxis: 'y3'
            }},
            {{
                x: predValues,
                y: gaussianValues,
                mode: 'lines',
                name: 'Gaussian',
                line: {{ color: '#d62728', width: 2.5 }},
                xaxis: 'x4',
                yaxis: 'y4'
            }}
        ];

        const subplotLayout = {{
            title: 'Individual Reward Functions',
            grid: {{ rows: 2, columns: 2, pattern: 'independent' }},
            annotations: [
                {{ text: 'L1 (Linear)', x: 0.225, y: 1, xref: 'paper', yref: 'paper', showarrow: false, font: {{ size: 14, color: '#1f77b4' }} }},
                {{ text: 'L2 (Quadratic)', x: 0.775, y: 1, xref: 'paper', yref: 'paper', showarrow: false, font: {{ size: 14, color: '#ff7f0e' }} }},
                {{ text: 'Laplace', x: 0.225, y: 0.45, xref: 'paper', yref: 'paper', showarrow: false, font: {{ size: 14, color: '#2ca02c' }} }},
                {{ text: 'Gaussian', x: 0.775, y: 0.45, xref: 'paper', yref: 'paper', showarrow: false, font: {{ size: 14, color: '#d62728' }} }}
            ],
            xaxis1: {{ title: 'Prediction', range: [1, 5], anchor: 'y1' }},
            yaxis1: {{ title: 'Reward', range: [0, 1.05], anchor: 'x1' }},
            xaxis2: {{ title: 'Prediction', range: [1, 5], anchor: 'y2' }},
            yaxis2: {{ title: 'Reward', range: [0, 1.05], anchor: 'x2' }},
            xaxis3: {{ title: 'Prediction', range: [1, 5], anchor: 'y3' }},
            yaxis3: {{ title: 'Reward', range: [0, 1.05], anchor: 'x3' }},
            xaxis4: {{ title: 'Prediction', range: [1, 5], anchor: 'y4' }},
            yaxis4: {{ title: 'Reward', range: [0, 1.05], anchor: 'x4' }},
            showlegend: false,
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            height: 700
        }};

        Plotly.newPlot('subplots', subplotTraces, subplotLayout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    output_file = 'reward_functions_comparison.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML visualization saved to: {output_file}")
    print(f"  Open this file in your web browser to view the interactive plots!")
    return output_file


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Interactive HTML Reward Function Visualization")
    print("=" * 60)
    print()
    
    try:
        output_file = create_html_visualization()
        print()
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Open {output_file} in your browser to view the plots.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


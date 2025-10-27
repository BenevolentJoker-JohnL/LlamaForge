#!/usr/bin/env python3
"""
Simple LlamaForge Training Dashboard
Shows training progress and system status
"""

from flask import Flask, render_template_string
import psutil
import torch
import os
import json
from pathlib import Path

app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>LlamaForge Training Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {
            background: #0a0e27;
            color: #00ff41;
            font-family: 'Courier New', monospace;
            padding: 20px;
            margin: 0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #00ff41;
            text-shadow: 0 0 10px #00ff41;
            border-bottom: 2px solid #00ff41;
            padding-bottom: 10px;
        }
        .card {
            background: #1a1f3a;
            border: 1px solid #00ff41;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 0 20px rgba(0,255,65,0.2);
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #0a3d1f;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .label {
            color: #00dd88;
        }
        .value {
            color: #00ff41;
            font-weight: bold;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #0a3d1f;
            border: 1px solid #00ff41;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff41, #00dd88);
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #0a0e27;
            font-weight: bold;
        }
        .status-online {
            color: #00ff41;
        }
        .status-offline {
            color: #ff4141;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ LLAMAFORGE TRAINING DASHBOARD ⚡</h1>

        <div class="card">
            <h2>System Resources</h2>
            <div class="metric">
                <span class="label">CPU Usage:</span>
                <span class="value">{{ cpu_percent }}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ cpu_percent }}%">{{ cpu_percent }}%</div>
            </div>

            <div class="metric">
                <span class="label">RAM Usage:</span>
                <span class="value">{{ ram_used_gb }} / {{ ram_total_gb }} GB ({{ ram_percent }}%)</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ ram_percent }}%">{{ ram_percent }}%</div>
            </div>

            <div class="metric">
                <span class="label">GPU:</span>
                <span class="value {{ 'status-online' if has_gpu else 'status-offline' }}">
                    {{ 'Available' if has_gpu else 'Not Detected' }}
                </span>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Training Status</h2>
                <div class="metric">
                    <span class="label">Active Jobs:</span>
                    <span class="value">{{ active_jobs }}</span>
                </div>
                <div class="metric">
                    <span class="label">Queued Jobs:</span>
                    <span class="value">{{ queued_jobs }}</span>
                </div>
                <div class="metric">
                    <span class="label">Completed:</span>
                    <span class="value">{{ completed_jobs }}</span>
                </div>
            </div>

            <div class="card">
                <h2>Quick Stats</h2>
                <div class="metric">
                    <span class="label">CPU Cores:</span>
                    <span class="value">{{ cpu_count }}</span>
                </div>
                <div class="metric">
                    <span class="label">Python Version:</span>
                    <span class="value">{{ python_version }}</span>
                </div>
                <div class="metric">
                    <span class="label">PyTorch Version:</span>
                    <span class="value">{{ torch_version }}</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Recent Activity</h2>
            <div class="metric">
                <span class="label">Last Update:</span>
                <span class="value">{{ last_update }}</span>
            </div>
            <div class="metric">
                <span class="label">Status:</span>
                <span class="value status-online">ONLINE - Auto-refresh every 5s</span>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    import sys
    from datetime import datetime

    # Get system stats
    cpu_percent = round(psutil.cpu_percent(interval=0.1), 1)
    cpu_count = psutil.cpu_count()

    mem = psutil.virtual_memory()
    ram_total_gb = round(mem.total / (1024**3), 1)
    ram_used_gb = round((mem.total - mem.available) / (1024**3), 1)
    ram_percent = round(mem.percent, 1)

    has_gpu = torch.cuda.is_available()

    # Training job stats (placeholder - would connect to actual job tracker)
    active_jobs = 0
    queued_jobs = 0
    completed_jobs = 0

    # Check for running training processes
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            if 'llamaforge' in ' '.join(proc.info['cmdline'] or []).lower():
                active_jobs += 1
        except:
            pass

    return render_template_string(
        DASHBOARD_HTML,
        cpu_percent=cpu_percent,
        cpu_count=cpu_count,
        ram_total_gb=ram_total_gb,
        ram_used_gb=ram_used_gb,
        ram_percent=ram_percent,
        has_gpu=has_gpu,
        active_jobs=active_jobs,
        queued_jobs=queued_jobs,
        completed_jobs=completed_jobs,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        torch_version=torch.__version__,
        last_update=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

if __name__ == '__main__':
    print("=" * 80)
    print(" " * 20 + "🚀 LLAMAFORGE DASHBOARD SERVER 🚀")
    print("=" * 80)
    print()
    print("  Dashboard URL: http://localhost:5000")
    print("  Auto-refresh: Every 5 seconds")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 80)
    print()

    app.run(host='0.0.0.0', port=5000, debug=False)

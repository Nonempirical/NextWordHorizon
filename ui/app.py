"""
Gradio application for Horizon Expansion.

This module defines the user interface for:
- Entering prompts and parameters
- Visualizing expanded horizon trees in 3D
- Displaying metrics (BF/entropy) per depth
- Interacting with expansion in real-time

The UI can run locally or in a Colab environment.
"""

import gradio as gr
import requests
import plotly.graph_objects as go
from typing import Optional, Tuple
import pandas as pd

DEFAULT_API_URL = "http://localhost:8000"


def _check_api_available(api_url: str) -> bool:
    """Check if the API server is available."""
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def explore_horizon(
    prompt: str,
    top_k: int,
    max_depth: int,
    model_backend: str,
    model_name: str,
    remote_base_url: str,
    api_url: str = DEFAULT_API_URL
) -> Tuple[go.Figure, go.Figure, pd.DataFrame]:
    """
    Calls the API to expand horizon and returns visualizations.
    
    Args:
        prompt: Input prompt
        top_k: Number of top candidates
        max_depth: Maximum depth
        model_backend: "local" or "remote"
        model_name: Model name for local backend
        remote_base_url: Base URL for remote backend
        api_url: URL to API server
        
    Returns:
        Tuple of (3D figure, 2D figure, data table)
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    payload = {
        "prompt": prompt.strip(),
        "top_k": int(top_k),
        "max_depth": int(max_depth),
        "model_backend": model_backend
    }
    
    if model_backend == "local":
        if not model_name:
            raise ValueError("model_name is required for local backend")
        payload["model_name"] = model_name
        payload["model_name"] = model_name
    elif model_backend == "remote":
        if not remote_base_url:
            raise ValueError("remote_base_url is required for remote backend")
        payload["remote_base_url"] = remote_base_url
    
    # Check if API is available first
    if not _check_api_available(api_url):
        raise ValueError(
            f"Cannot connect to API server at {api_url}. "
            "Please make sure the API server is running. "
            "Check the separate API server window or run: "
            "uvicorn api.server:app --reload --host 0.0.0.0 --port 8000"
        )
    
    # Call API
    try:
        response = requests.post(
            f"{api_url}/expand_horizon",
            json=payload,
            timeout=300  # 5 minute timeout for large models
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError as e:
        raise ValueError(
            f"Cannot connect to API server at {api_url}. "
            "The server may not be running or may have crashed. "
            "Please check the API server window."
        )
    except requests.exceptions.Timeout as e:
        raise ValueError(
            f"Request to API server timed out. "
            "The server may be overloaded or processing a large request."
        )
    except requests.exceptions.RequestException as e:
        raise ValueError(f"API request failed: {str(e)}")
    
    # Build 3D figure
    fig_3d = _create_3d_plot(data)
    
    # Build 2D figure (BF/entropy per depth)
    fig_2d = _create_2d_plot(data)
    
    # Build data table
    df = _create_data_table(data)
    
    return fig_3d, fig_2d, df


def _create_3d_plot(data: dict) -> go.Figure:
    """Creates a 3D Plotly figure of the horizon tree."""
    if not isinstance(data, dict) or "nodes" not in data or "edges" not in data:
        fig = go.Figure()
        fig.add_annotation(
            text="Invalid data format",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    nodes_with_proj = [n for n in nodes if n.get("proj") is not None and len(n.get("proj", [])) >= 3]
    
    if len(nodes_with_proj) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No 3D projection available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    x_coords = [n["proj"][0] for n in nodes_with_proj]
    y_coords = [n["proj"][1] for n in nodes_with_proj]
    z_coords = [n["proj"][2] for n in nodes_with_proj]
    depths = [n["depth"] for n in nodes_with_proj]
    cumulative_probs = [n["cumulative_prob"] for n in nodes_with_proj]
    node_ids = [n["id"] for n in nodes_with_proj]
    token_texts = [n.get("token_text", "") or "" for n in nodes_with_proj]
    
    # Create node dictionary for quick lookup
    node_dict = {n["id"]: n for n in nodes_with_proj}
    
    # Create figure
    fig = go.Figure()
    
    # Add edges (lines between parent and child)
    for edge in edges:
        source_id, target_id = edge[0], edge[1]
        if source_id in node_dict and target_id in node_dict:
            source = node_dict[source_id]
            target = node_dict[target_id]
            if source["proj"] and target["proj"]:
                fig.add_trace(go.Scatter3d(
                    x=[source["proj"][0], target["proj"][0]],
                    y=[source["proj"][1], target["proj"][1]],
                    z=[source["proj"][2], target["proj"][2]],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Add nodes (scatter3d)
    # Size based on log(cumulative_prob) for better visual distinction
    import numpy as np
    sizes = [max(5, 20 * np.log10(max(prob, 1e-6) + 1)) for prob in cumulative_probs]
    
    # Color coding by depth
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=depths,
            colorscale='Viridis',
            colorbar=dict(title="Depth"),
            line=dict(width=1, color='black')
        ),
        text=token_texts,
        textposition="middle center",
        textfont=dict(size=8),
        hovertemplate=(
            '<b>%{text}</b><br>'
            'ID: %{customdata[0]}<br>'
            'Depth: %{customdata[1]}<br>'
            'Cumulative Prob: %{customdata[2]:.4f}<extra></extra>'
        ),
        customdata=[[nid, d, p] for nid, d, p in zip(node_ids, depths, cumulative_probs)],
        name="Nodes"
    ))
    
    fig.update_layout(
        title="3D Horizon Tree Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube'
        ),
        height=600
    )
    
    return fig


def _create_2d_plot(data: dict) -> go.Figure:
    """
    Creates a 2D Plotly figure with BF and entropy per depth.
    
    Args:
        data: JSON data from API
        
    Returns:
        Plotly Figure with line chart
    """
    depth_stats = data.get("depth_stats", {})
    
    if len(depth_stats) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No depth statistics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Extract data
    depths = sorted([int(d) for d in depth_stats.keys()])
    try:
        entropies = [depth_stats[str(d)].get("entropy", 0.0) for d in depths]
        branching_factors = [depth_stats[str(d)].get("branching_factor", 0.0) for d in depths]
    except (KeyError, TypeError):
        fig = go.Figure()
        fig.add_annotation(
            text="Invalid depth statistics data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create figure with two y-axes
    fig = go.Figure()
    
    # Entropy (left y-axis)
    fig.add_trace(go.Scatter(
        x=depths,
        y=entropies,
        mode='lines+markers',
        name='Entropy',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Branching Factor (right y-axis)
    fig.add_trace(go.Scatter(
        x=depths,
        y=branching_factors,
        mode='lines+markers',
        name='Branching Factor',
        line=dict(color='red', width=2),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Entropy and Branching Factor per Depth",
        xaxis=dict(title="Depth", dtick=1),
        yaxis=dict(title="Entropy (bits)", side='left', color='blue'),
        yaxis2=dict(
            title="Branching Factor",
            side='right',
            color='red',
            overlaying='y'
        ),
        height=400,
        hovermode='x unified'
    )
    
    return fig


def _create_data_table(data: dict) -> pd.DataFrame:
    """
    Creates a data table with tokens per depth.
    
    Args:
        data: JSON data from API
        
    Returns:
        DataFrame with columns: depth, token_text, local_prob, cumulative_prob
    """
    if not isinstance(data, dict) or "nodes" not in data:
        return pd.DataFrame(columns=["depth", "token_text", "local_prob", "cumulative_prob"])
    
    nodes = data.get("nodes", [])
    
    # Filter out root node
    nodes_with_tokens = [
        n for n in nodes
        if n.get("token_text") is not None and n.get("depth", 0) > 0
    ]
    
    if len(nodes_with_tokens) == 0:
        return pd.DataFrame(columns=["depth", "token_text", "local_prob", "cumulative_prob"])
    
    # Sort by depth and cumulative_prob (descending)
    nodes_sorted = sorted(
        nodes_with_tokens,
        key=lambda n: (n.get("depth", 0), -n.get("cumulative_prob", 0))
    )
    
    # Build DataFrame
    df = pd.DataFrame([
        {
            "depth": n.get("depth", 0),
            "token_text": n.get("token_text", ""),
            "local_prob": n.get("local_prob", 0.0),
            "cumulative_prob": n.get("cumulative_prob", 0.0)
        }
        for n in nodes_sorted
    ])
    
    return df


def create_ui(api_url: str = DEFAULT_API_URL):
    """
    Creates and returns the Gradio interface.
    
    Args:
        api_url: URL to API server
        
    Returns:
        Gradio Interface object
    """
    with gr.Blocks(title="Next Word Horizon") as interface:
        gr.Markdown("# Next Word Horizon")
        gr.Markdown("Probabilistic Horizon Expansion for next-word prediction")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Input")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                    value="The dog"
                )
                
                with gr.Row():
                    top_k = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Top-K candidates"
                    )
                    max_depth = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Maximum depth"
                    )
                
                model_backend = gr.Dropdown(
                    choices=["local", "remote"],
                    value="local",
                    label="Model Backend"
                )
                
                model_name = gr.Textbox(
                    label="Model Name (for local backend)",
                    placeholder="e.g. ./models/qwen2.5-1.5b, gpt2, Qwen/Qwen2.5-1.5B-Instruct",
                    value="./models/qwen2.5-1.5b"
                )
                
                remote_base_url = gr.Textbox(
                    label="Remote Base URL (for remote backend)",
                    placeholder="e.g. http://localhost:8000",
                    value=""
                )
                
                api_url_input = gr.Textbox(
                    label="API URL",
                    value=api_url,
                    placeholder="http://localhost:8000"
                )
                
                explore_btn = gr.Button("Run / Explore", variant="primary")
            
            with gr.Column():
                gr.Markdown("## Output")
                
                plot_3d = gr.Plot(
                    label="3D Horizon Tree",
                    show_label=True
                )
                
                plot_2d = gr.Plot(
                    label="Entropy & Branching Factor per Depth",
                    show_label=True
                )
                
                data_table = gr.Dataframe(
                    label="Tokens per Depth",
                    headers=["depth", "token_text", "local_prob", "cumulative_prob"],
                    wrap=True
                )
        
        # Connect function to button
        explore_btn.click(
            fn=lambda p, tk, md, mb, mn, rbu, api: explore_horizon(
                p, tk, md, mb, mn, rbu, api
            ),
            inputs=[prompt, top_k, max_depth, model_backend, model_name, remote_base_url, api_url_input],
            outputs=[plot_3d, plot_2d, data_table]
        )
    
    return interface


def main():
    """Main function to start the UI."""
    import os
    api_url = os.getenv("HORIZON_API_URL", DEFAULT_API_URL)
    
    interface = create_ui(api_url=api_url)
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()


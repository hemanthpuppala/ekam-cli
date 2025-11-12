"""HTTP server for custom llama.cpp CLI server.

This module provides a lightweight Flask HTTP server that routes requests
to slot-based processors with isolated mtmd_context.
"""

import sys
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Flask, request, Response, jsonify
from loguru import logger


def create_app(
    manager,
    host: str = "127.0.0.1",
    port: int = 8080
) -> Flask:
    """Create Flask app for custom CLI server.

    Args:
        manager: LlamaCLIServerManager instance
        host: Server host
        port: Server port

    Returns:
        Flask app instance
    """
    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok"}), 200

    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        """Chat completions endpoint (OpenAI-compatible)."""
        try:
            payload = request.get_json()

            # Extract parameters
            messages = payload.get('messages', [])
            max_tokens = payload.get('max_tokens', 512)
            temperature = payload.get('temperature', 0.7)
            top_p = payload.get('top_p', 0.9)
            stream = payload.get('stream', True)
            session_id = payload.get('session_id')  # For session affinity

            # Get or allocate slot
            slot = manager._get_slot_for_session(session_id) if session_id else manager._get_free_slot()

            if slot is None:
                return jsonify({"error": "No available slots"}), 503

            # Process request in slot
            result = manager._process_request_in_slot(
                slot=slot,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                **{k: v for k, v in payload.items() if k not in ['messages', 'max_tokens', 'temperature', 'top_p', 'stream', 'session_id']}
            )

            if stream:
                return Response(result, mimetype='text/event-stream')
            else:
                return jsonify(result)

        except Exception as e:
            logger.error(f"Error in chat_completions: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    return app


def run_server(
    manager_state: Dict[str, Any],
    host: str,
    port: int
):
    """Run HTTP server in subprocess.

    Args:
        manager_state: Serialized manager state
        host: Server host
        port: Server port
    """
    # Note: This function would run in subprocess
    # For now, we'll use threading approach instead
    pass

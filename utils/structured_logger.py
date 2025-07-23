"""
Structured logging utility for JamPacked autonomous agents
Provides trace IDs, structured JSON output, and observability
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager
from threading import local

# Thread-local storage for trace context
_trace_context = local()


class StructuredLogger:
    """
    Structured logger with trace ID support and JSON output
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Add JSON formatter if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = JSONFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _get_trace_context(self) -> Dict[str, Any]:
        """Get current trace context"""
        return getattr(_trace_context, 'context', {})
    
    def _log_structured(self, level: str, message: str, **kwargs):
        """Log with structured data"""
        trace_context = self._get_trace_context()
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "message": message,
            "trace_id": trace_context.get("trace_id"),
            "goal_id": trace_context.get("goal_id"),
            "agent_id": trace_context.get("agent_id"),
            "parent_id": trace_context.get("parent_id"),
            "component": trace_context.get("component", "unknown"),
            **kwargs
        }
        
        # Remove None values
        log_data = {k: v for k, v in log_data.items() if v is not None}
        
        getattr(self.logger, level.lower())(json.dumps(log_data))
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_structured("INFO", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_structured("DEBUG", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_structured("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_structured("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_structured("CRITICAL", message, **kwargs)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        # If record.msg is already JSON (from structured logger), return as-is
        if hasattr(record, 'msg') and record.msg.startswith('{'):
            return record.msg
        
        # Otherwise, format as simple JSON
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        return json.dumps(log_data)


class TraceManager:
    """Manages trace context for distributed tracing"""
    
    @staticmethod
    def new_trace_id() -> str:
        """Generate a new trace ID"""
        return str(uuid.uuid4())
    
    @staticmethod
    def set_trace_context(trace_id: str, **context):
        """Set trace context for current thread"""
        if not hasattr(_trace_context, 'context'):
            _trace_context.context = {}
        
        _trace_context.context.update({
            "trace_id": trace_id,
            **context
        })
    
    @staticmethod
    def get_trace_id() -> Optional[str]:
        """Get current trace ID"""
        context = getattr(_trace_context, 'context', {})
        return context.get("trace_id")
    
    @staticmethod
    def clear_trace_context():
        """Clear trace context"""
        if hasattr(_trace_context, 'context'):
            _trace_context.context.clear()


@contextmanager
def trace_context(trace_id: Optional[str] = None, **context):
    """
    Context manager for trace context
    
    Usage:
        with trace_context(component="autonomous_engine", goal_id="goal_123"):
            logger.info("Processing task")
    """
    if trace_id is None:
        trace_id = TraceManager.new_trace_id()
    
    # Save existing context
    old_context = getattr(_trace_context, 'context', {}).copy()
    
    try:
        TraceManager.set_trace_context(trace_id, **context)
        yield trace_id
    finally:
        # Restore old context
        _trace_context.context = old_context


class PerformanceTimer:
    """Context manager for timing operations with structured logging"""
    
    def __init__(self, logger: StructuredLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                duration_ms=round(duration_ms, 2),
                status="success",
                **self.context
            )
        else:
            self.logger.error(
                f"Failed {self.operation}",
                duration_ms=round(duration_ms, 2),
                status="error",
                error_type=exc_type.__name__ if exc_type else None,
                error_message=str(exc_val) if exc_val else None,
                **self.context
            )


class CostTracker:
    """Track and log costs for operations"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.costs = {}
    
    def track_cost(self, operation: str, cost_usd: float, tokens_used: int = 0, **context):
        """Track cost for an operation"""
        cost_data = {
            "operation": operation,
            "cost_usd": round(cost_usd, 4),
            "tokens_used": tokens_used,
            "timestamp": datetime.utcnow().isoformat(),
            **context
        }
        
        # Store for aggregation
        if operation not in self.costs:
            self.costs[operation] = []
        self.costs[operation].append(cost_data)
        
        # Log immediately
        self.logger.info(
            f"Cost tracked for {operation}",
            **cost_data
        )
    
    def get_total_cost(self, operation: Optional[str] = None) -> float:
        """Get total cost for operation or all operations"""
        if operation:
            return sum(c["cost_usd"] for c in self.costs.get(operation, []))
        else:
            return sum(
                sum(c["cost_usd"] for c in costs)
                for costs in self.costs.values()
            )


# Global instances
def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)


# Example usage:
if __name__ == "__main__":
    logger = get_logger("test")
    
    with trace_context(component="test", goal_id="test_goal"):
        logger.info("Test message", custom_field="value")
        
        with PerformanceTimer(logger, "test_operation", subtask_id="subtask_1"):
            time.sleep(0.1)
        
        cost_tracker = CostTracker(logger)
        cost_tracker.track_cost("test_operation", 0.05, tokens_used=100)
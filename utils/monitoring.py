"""
Performance Monitoring - Track cycle times, memory usage, and system health

PERFORMANCE: This module helps identify bottlenecks and resource issues
in long-running trading processes.
"""
import time
import logging
import functools
import tracemalloc
import gc
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class CycleMetrics:
    """Metrics for a single execution cycle"""
    cycle_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    signals_generated: int = 0
    signals_executed: int = 0
    memory_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    stages: Dict[str, float] = field(default_factory=dict)  # stage_name -> duration_ms


class PerformanceMonitor:
    """
    Monitor performance metrics for the trading bot.
    
    Tracks:
    - Cycle execution times
    - Memory usage over time
    - Stage-level timing (data fetch, signal gen, execution)
    - Error rates
    
    Alerts when:
    - Cycle time exceeds threshold (can't keep up with 5-min cycles)
    - Memory usage grows continuously (leak)
    - Error rate too high
    """
    
    # Thresholds
    MAX_CYCLE_TIME_MS = 240_000  # 4 minutes (must finish before next 5-min cycle)
    MEMORY_GROWTH_ALERT_MB = 100  # Alert if memory grows by this much
    MAX_ERROR_RATE = 0.10  # 10% error rate triggers alert
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.cycles: deque = deque(maxlen=history_size)
        self.current_cycle: Optional[CycleMetrics] = None
        self.cycle_counter = 0
        self._lock = threading.Lock()
        
        # Memory tracking
        self.initial_memory_mb: Optional[float] = None
        self.peak_memory_mb: float = 0.0
        
        # Stage timing context
        self._stage_start: Optional[float] = None
        self._current_stage: Optional[str] = None
        
        # Start memory tracking
        try:
            tracemalloc.start()
            self.initial_memory_mb = self._get_memory_mb()
            logger.info(f"Performance monitor started. Initial memory: {self.initial_memory_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"Could not start tracemalloc: {e}")
    
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            # Try tracemalloc first (more accurate)
            current, peak = tracemalloc.get_traced_memory()
            return current / (1024 * 1024)
        except:
            pass
        
        try:
            # Fall back to process memory
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # Convert KB to MB
        except:
            pass
        
        return 0.0
    
    def start_cycle(self) -> int:
        """Start a new execution cycle"""
        with self._lock:
            self.cycle_counter += 1
            self.current_cycle = CycleMetrics(
                cycle_id=self.cycle_counter,
                start_time=datetime.utcnow(),
                memory_mb=self._get_memory_mb()
            )
            return self.cycle_counter
    
    def start_stage(self, stage_name: str):
        """Start timing a stage within the cycle"""
        self._stage_start = time.perf_counter() * 1000
        self._current_stage = stage_name
    
    def end_stage(self, stage_name: str = None):
        """End timing a stage"""
        if self._stage_start is None:
            return
        
        stage = stage_name or self._current_stage or "unknown"
        duration_ms = time.perf_counter() * 1000 - self._stage_start
        
        if self.current_cycle:
            self.current_cycle.stages[stage] = duration_ms
        
        self._stage_start = None
        self._current_stage = None
    
    def record_signals(self, generated: int = 0, executed: int = 0):
        """Record signal counts for current cycle"""
        if self.current_cycle:
            self.current_cycle.signals_generated += generated
            self.current_cycle.signals_executed += executed
    
    def record_error(self, error: str):
        """Record an error in current cycle"""
        if self.current_cycle:
            self.current_cycle.errors.append(error[:500])  # Truncate long errors
    
    def end_cycle(self) -> Optional[CycleMetrics]:
        """End current cycle and return metrics"""
        with self._lock:
            if not self.current_cycle:
                return None
            
            cycle = self.current_cycle
            cycle.end_time = datetime.utcnow()
            cycle.duration_ms = (cycle.end_time - cycle.start_time).total_seconds() * 1000
            cycle.memory_mb = self._get_memory_mb()
            
            # Track peak memory
            if cycle.memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = cycle.memory_mb
            
            self.cycles.append(cycle)
            self.current_cycle = None
            
            # Check for issues
            self._check_alerts(cycle)
            
            # Log cycle summary
            logger.info(
                f"Cycle #{cycle.cycle_id} completed in {cycle.duration_ms:.0f}ms | "
                f"Signals: {cycle.signals_generated} gen, {cycle.signals_executed} exec | "
                f"Memory: {cycle.memory_mb:.1f} MB"
            )
            
            # Log stage breakdown if slow
            if cycle.duration_ms > 10000:  # 10 seconds
                stage_summary = ", ".join(f"{k}: {v:.0f}ms" for k, v in cycle.stages.items())
                logger.info(f"  Stage breakdown: {stage_summary}")
            
            return cycle
    
    def _check_alerts(self, cycle: CycleMetrics):
        """Check for performance alerts"""
        # Slow cycle
        if cycle.duration_ms > self.MAX_CYCLE_TIME_MS:
            logger.warning(
                f"⚠️ SLOW CYCLE: #{cycle.cycle_id} took {cycle.duration_ms/1000:.1f}s "
                f"(max: {self.MAX_CYCLE_TIME_MS/1000:.0f}s)"
            )
        
        # Memory growth
        if self.initial_memory_mb:
            growth = cycle.memory_mb - self.initial_memory_mb
            if growth > self.MEMORY_GROWTH_ALERT_MB:
                logger.warning(
                    f"⚠️ MEMORY GROWTH: {growth:.1f} MB since start "
                    f"(current: {cycle.memory_mb:.1f} MB)"
                )
        
        # Error rate
        if len(self.cycles) >= 10:
            recent = list(self.cycles)[-10:]
            error_cycles = sum(1 for c in recent if c.errors)
            error_rate = error_cycles / len(recent)
            if error_rate > self.MAX_ERROR_RATE:
                logger.warning(
                    f"⚠️ HIGH ERROR RATE: {error_rate:.0%} of recent cycles had errors"
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.cycles:
            return {'status': 'no_data'}
        
        cycles_list = list(self.cycles)
        durations = [c.duration_ms for c in cycles_list]
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        p50_idx = len(sorted_durations) // 2
        p95_idx = int(len(sorted_durations) * 0.95)
        p99_idx = int(len(sorted_durations) * 0.99)
        
        # Error stats
        error_cycles = sum(1 for c in cycles_list if c.errors)
        
        return {
            'total_cycles': len(cycles_list),
            'cycle_time_ms': {
                'avg': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'p50': sorted_durations[p50_idx] if sorted_durations else 0,
                'p95': sorted_durations[p95_idx] if len(sorted_durations) > 20 else 0,
                'p99': sorted_durations[p99_idx] if len(sorted_durations) > 100 else 0,
            },
            'memory_mb': {
                'initial': self.initial_memory_mb,
                'current': self._get_memory_mb(),
                'peak': self.peak_memory_mb,
                'growth': self._get_memory_mb() - (self.initial_memory_mb or 0),
            },
            'signals': {
                'total_generated': sum(c.signals_generated for c in cycles_list),
                'total_executed': sum(c.signals_executed for c in cycles_list),
                'avg_per_cycle': sum(c.signals_generated for c in cycles_list) / len(cycles_list),
            },
            'errors': {
                'total_error_cycles': error_cycles,
                'error_rate': error_cycles / len(cycles_list),
            },
            'last_10_cycles': [
                {
                    'cycle_id': c.cycle_id,
                    'duration_ms': c.duration_ms,
                    'signals': c.signals_generated,
                    'errors': len(c.errors),
                }
                for c in list(self.cycles)[-10:]
            ]
        }
    
    def force_gc(self) -> float:
        """Force garbage collection and return memory freed (MB)"""
        before = self._get_memory_mb()
        gc.collect()
        after = self._get_memory_mb()
        freed = before - after
        
        if freed > 1:  # Only log if significant
            logger.info(f"GC freed {freed:.1f} MB")
        
        return freed


# Global monitor instance
_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def timed_stage(stage_name: str):
    """Decorator to time a function as a stage"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            monitor.start_stage(stage_name)
            try:
                return func(*args, **kwargs)
            finally:
                monitor.end_stage(stage_name)
        return wrapper
    return decorator


def timed(func: Callable) -> Callable:
    """Simple timing decorator that logs execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            duration = (time.perf_counter() - start) * 1000
            if duration > 1000:  # Log if > 1 second
                logger.debug(f"{func.__name__} took {duration:.0f}ms")
    return wrapper

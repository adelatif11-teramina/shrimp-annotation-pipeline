"""
Monitoring and Metrics Collection
Provides application metrics, health monitoring, and alerting
"""

import time
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import logging

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# Sentry error tracking (optional)
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
    HAS_SENTRY = True
except ImportError:
    HAS_SENTRY = False

logger = logging.getLogger(__name__)

@dataclass
class MetricData:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram

@dataclass
class HealthStatus:
    """System health status"""
    status: str  # healthy, degraded, unhealthy
    components: Dict[str, bool]
    metrics: Dict[str, float]
    timestamp: datetime
    message: Optional[str] = None

class MetricsCollector:
    """Collects and stores application metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Prometheus registry (if available)
        if HAS_PROMETHEUS:
            self.registry = CollectorRegistry()
            self.prom_counters: Dict[str, Counter] = {}
            self.prom_gauges: Dict[str, Gauge] = {}
            self.prom_histograms: Dict[str, Histogram] = {}
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        labels = labels or {}
        
        # Internal tracking
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        self.counters[key] += value
        
        # Add to history
        metric = MetricData(
            name=name,
            value=self.counters[key],
            timestamp=datetime.utcnow(),
            labels=labels,
            metric_type="counter"
        )
        self.metrics_history[name].append(metric)
        
        # Prometheus (if available)
        if HAS_PROMETHEUS:
            if name not in self.prom_counters:
                self.prom_counters[name] = Counter(
                    name, f"Counter: {name}", 
                    labelnames=list(labels.keys()) if labels else [],
                    registry=self.registry
                )
            
            if labels:
                self.prom_counters[name].labels(**labels).inc(value)
            else:
                self.prom_counters[name].inc(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        labels = labels or {}
        
        # Internal tracking
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        self.gauges[key] = value
        
        # Add to history
        metric = MetricData(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels,
            metric_type="gauge"
        )
        self.metrics_history[name].append(metric)
        
        # Prometheus (if available)
        if HAS_PROMETHEUS:
            if name not in self.prom_gauges:
                self.prom_gauges[name] = Gauge(
                    name, f"Gauge: {name}",
                    labelnames=list(labels.keys()) if labels else [],
                    registry=self.registry
                )
            
            if labels:
                self.prom_gauges[name].labels(**labels).set(value)
            else:
                self.prom_gauges[name].set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram observation"""
        labels = labels or {}
        
        # Internal tracking
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        self.histograms[key].append(value)
        
        # Keep only recent values
        if len(self.histograms[key]) > self.max_history:
            self.histograms[key] = self.histograms[key][-self.max_history:]
        
        # Add to history
        metric = MetricData(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels,
            metric_type="histogram"
        )
        self.metrics_history[name].append(metric)
        
        # Prometheus (if available)
        if HAS_PROMETHEUS:
            if name not in self.prom_histograms:
                self.prom_histograms[name] = Histogram(
                    name, f"Histogram: {name}",
                    labelnames=list(labels.keys()) if labels else [],
                    registry=self.registry
                )
            
            if labels:
                self.prom_histograms[name].labels(**labels).observe(value)
            else:
                self.prom_histograms[name].observe(value)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {}
        }
        
        # Calculate histogram statistics
        for key, values in self.histograms.items():
            if values:
                summary["histograms"][key] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p50": sorted(values)[len(values) // 2],
                    "p95": sorted(values)[int(len(values) * 0.95)],
                    "p99": sorted(values)[int(len(values) * 0.99)]
                }
        
        return summary
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        if not HAS_PROMETHEUS:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')

class HealthChecker:
    """Monitors system health and component status"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks: Dict[str, Callable] = {}
        self.component_status: Dict[str, bool] = {}
        self.last_check: Optional[datetime] = None
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function"""
        self.health_checks[name] = check_func
    
    async def check_system_health(self) -> HealthStatus:
        """Perform comprehensive health check"""
        now = datetime.utcnow()
        components = {}
        system_metrics = {}
        
        # Check registered components
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                components[name] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                components[name] = False
        
        # System metrics
        try:
            # CPU usage - avoid interval=1 which can cause floating point exceptions
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                # Validate CPU percentage to avoid floating point errors
                if cpu_percent is not None and 0 <= cpu_percent <= 100:
                    system_metrics["cpu_usage"] = cpu_percent
                    self.metrics.set_gauge("system_cpu_usage", cpu_percent)
                else:
                    # Fallback to non-blocking call
                    cpu_percent = psutil.cpu_percent(interval=None)
                    if cpu_percent is not None and 0 <= cpu_percent <= 100:
                        system_metrics["cpu_usage"] = cpu_percent
                        self.metrics.set_gauge("system_cpu_usage", cpu_percent)
            except (ZeroDivisionError, OverflowError, OSError) as e:
                logger.warning(f"Failed to get CPU usage: {e}")
                system_metrics["cpu_usage"] = 0.0
            
            # Memory usage
            try:
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                if memory_percent is not None and 0 <= memory_percent <= 100:
                    system_metrics["memory_usage"] = memory_percent
                    self.metrics.set_gauge("system_memory_usage", memory_percent)
            except (ZeroDivisionError, OverflowError, OSError) as e:
                logger.warning(f"Failed to get memory usage: {e}")
                system_metrics["memory_usage"] = 0.0
            
            # Disk usage
            try:
                disk = psutil.disk_usage('/')
                if disk.total > 0:  # Avoid division by zero
                    disk_percent = (disk.used / disk.total) * 100
                    if 0 <= disk_percent <= 100:
                        system_metrics["disk_usage"] = disk_percent
                        self.metrics.set_gauge("system_disk_usage", disk_percent)
            except (ZeroDivisionError, OverflowError, OSError) as e:
                logger.warning(f"Failed to get disk usage: {e}")
                system_metrics["disk_usage"] = 0.0
            
            # Load average (Unix only)
            try:
                load_avg = psutil.getloadavg()
                if load_avg and len(load_avg) > 0 and load_avg[0] is not None:
                    system_metrics["load_average"] = load_avg[0]
                    self.metrics.set_gauge("system_load_average", load_avg[0])
            except (AttributeError, OSError):
                # Windows doesn't have load average or other OS errors
                pass
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
        
        # Determine overall status
        healthy_components = sum(1 for status in components.values() if status)
        total_components = len(components)
        
        if total_components == 0:
            status = "healthy"
        elif healthy_components == total_components:
            status = "healthy"
        elif healthy_components > total_components / 2:
            status = "degraded"
        else:
            status = "unhealthy"
        
        # Check critical thresholds
        if system_metrics.get("cpu_usage", 0) > 90:
            status = "degraded" if status == "healthy" else status
        if system_metrics.get("memory_usage", 0) > 90:
            status = "degraded" if status == "healthy" else status
        if system_metrics.get("disk_usage", 0) > 95:
            status = "unhealthy"
        
        self.last_check = now
        
        return HealthStatus(
            status=status,
            components=components,
            metrics=system_metrics,
            timestamp=now,
            message=f"System is {status}"
        )

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: deque = deque(maxlen=1000)
    
    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[], bool],
        message: str,
        severity: str = "warning",
        cooldown: int = 300  # 5 minutes
    ):
        """Add an alert rule"""
        self.alert_rules.append({
            "name": name,
            "condition": condition,
            "message": message,
            "severity": severity,
            "cooldown": cooldown,
            "last_triggered": None
        })
    
    async def check_alerts(self):
        """Check all alert rules and trigger alerts"""
        now = datetime.utcnow()
        
        for rule in self.alert_rules:
            try:
                # Check condition
                should_alert = await rule["condition"]() if asyncio.iscoroutinefunction(rule["condition"]) else rule["condition"]()
                
                alert_key = rule["name"]
                
                if should_alert:
                    # Check cooldown
                    if (rule["last_triggered"] and 
                        (now - rule["last_triggered"]).total_seconds() < rule["cooldown"]):
                        continue
                    
                    # Trigger alert
                    await self._trigger_alert(rule, now)
                    rule["last_triggered"] = now
                    
                else:
                    # Clear alert if it was active
                    if alert_key in self.active_alerts:
                        await self._clear_alert(alert_key, now)
                        
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    async def _trigger_alert(self, rule: Dict[str, Any], timestamp: datetime):
        """Trigger an alert"""
        alert_key = rule["name"]
        
        alert_data = {
            "name": rule["name"],
            "message": rule["message"],
            "severity": rule["severity"],
            "triggered_at": timestamp,
            "status": "active"
        }
        
        self.active_alerts[alert_key] = alert_data
        self.alert_history.append({**alert_data, "action": "triggered"})
        
        # Log the alert
        logger.warning(f"ALERT TRIGGERED: {rule['name']} - {rule['message']}")
        
        # Increment metrics
        self.metrics.increment_counter(
            "alerts_triggered_total",
            labels={"alert": rule["name"], "severity": rule["severity"]}
        )
        
        # Send notifications (would integrate with external services)
        await self._send_notification(alert_data)
    
    async def _clear_alert(self, alert_key: str, timestamp: datetime):
        """Clear an active alert"""
        if alert_key in self.active_alerts:
            alert_data = self.active_alerts[alert_key].copy()
            alert_data["cleared_at"] = timestamp
            alert_data["status"] = "cleared"
            
            self.alert_history.append({**alert_data, "action": "cleared"})
            del self.active_alerts[alert_key]
            
            logger.info(f"ALERT CLEARED: {alert_key}")
            
            self.metrics.increment_counter(
                "alerts_cleared_total",
                labels={"alert": alert_key}
            )
    
    async def _send_notification(self, alert_data: Dict[str, Any]):
        """Send alert notification (placeholder for external integrations)"""
        # This would integrate with:
        # - Email services
        # - Slack/Discord webhooks
        # - PagerDuty
        # - SMS services
        # etc.
        
        # For now, just log
        logger.warning(f"📢 ALERT NOTIFICATION: {alert_data}")

class ApplicationMonitor:
    """Main monitoring orchestrator"""
    
    def __init__(self, enable_prometheus: bool = True, enable_sentry: bool = False):
        self.metrics = MetricsCollector()
        self.health_checker = HealthChecker(self.metrics)
        self.alert_manager = AlertManager(self.metrics)
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Setup Sentry if enabled
        if enable_sentry and HAS_SENTRY:
            self._setup_sentry()
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_sentry(self):
        """Initialize Sentry error tracking"""
        sentry_dsn = os.getenv("SENTRY_DSN")
        if sentry_dsn:
            sentry_sdk.init(
                dsn=sentry_dsn,
                integrations=[
                    FastApiIntegration(),
                    SqlalchemyIntegration(),
                ],
                traces_sample_rate=0.1,
                environment=os.getenv("ENVIRONMENT", "development")
            )
            logger.info("Sentry error tracking initialized")
    
    def _register_default_health_checks(self):
        """Register default system health checks"""
        
        def check_memory():
            try:
                memory = psutil.virtual_memory()
                return memory.percent < 95 if memory.percent is not None else True
            except Exception:
                return True  # Assume healthy if check fails
        
        def check_disk():
            try:
                disk = psutil.disk_usage('/')
                return (disk.used / disk.total) * 100 < 98 if disk.total > 0 else True
            except Exception:
                return True  # Assume healthy if check fails
        
        def check_cpu():
            try:
                cpu_usage = psutil.cpu_percent(interval=0.1)
                return cpu_usage < 95 if cpu_usage is not None else True
            except Exception:
                return True  # Assume healthy if check fails
        
        self.health_checker.register_health_check("memory", check_memory)
        self.health_checker.register_health_check("disk", check_disk)
        self.health_checker.register_health_check("cpu", check_cpu)
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        
        def high_memory_usage():
            try:
                memory = psutil.virtual_memory()
                return memory.percent > 90 if memory.percent is not None else False
            except Exception:
                return False
        
        def high_disk_usage():
            try:
                disk = psutil.disk_usage('/')
                return (disk.used / disk.total) * 100 > 95 if disk.total > 0 else False
            except Exception:
                return False
        
        def high_cpu_usage():
            try:
                cpu_usage = psutil.cpu_percent(interval=0.1)
                return cpu_usage > 95 if cpu_usage is not None else False
            except Exception:
                return False
        
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            high_memory_usage,
            "Memory usage is above 90%",
            "warning",
            cooldown=600  # 10 minutes
        )
        
        self.alert_manager.add_alert_rule(
            "high_disk_usage", 
            high_disk_usage,
            "Disk usage is above 95%",
            "critical",
            cooldown=300  # 5 minutes
        )
        
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            high_cpu_usage,
            "CPU usage is above 95%",
            "warning",
            cooldown=600  # 10 minutes
        )
    
    async def start_monitoring(self, interval: int = 60):
        """Start the monitoring loop"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info(f"Started monitoring with {interval}s interval")
    
    async def stop_monitoring(self):
        """Stop the monitoring loop"""
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.is_running = False
        logger.info("Stopped monitoring")
    
    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        try:
            while self.is_running:
                try:
                    # Check system health
                    health_status = await self.health_checker.check_system_health()
                    self.metrics.set_gauge("system_health_score", 
                                          1.0 if health_status.status == "healthy" else 
                                          0.5 if health_status.status == "degraded" else 0.0)
                    
                    # Check alerts
                    await self.alert_manager.check_alerts()
                    
                    # Update active alerts count
                    self.metrics.set_gauge("active_alerts_count", len(self.alert_manager.active_alerts))
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations"""
        start_time = time.time()
        
        try:
            yield
            # Success
            self.metrics.increment_counter(f"{operation_name}_total", labels={"status": "success"})
            
        except Exception as e:
            # Error
            self.metrics.increment_counter(f"{operation_name}_total", labels={"status": "error"})
            self.metrics.increment_counter(f"{operation_name}_errors_total", 
                                         labels={"error_type": type(e).__name__})
            raise
            
        finally:
            # Duration
            duration = time.time() - start_time
            self.metrics.observe_histogram(f"{operation_name}_duration_seconds", duration)

# Global monitor instance
monitor = ApplicationMonitor()

# Export commonly used items
__all__ = [
    "MetricData",
    "HealthStatus", 
    "MetricsCollector",
    "HealthChecker",
    "AlertManager",
    "ApplicationMonitor",
    "monitor"
]
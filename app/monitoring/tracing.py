"""Optional OpenTelemetry tracing that deliberately excludes conversation content."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:  # Keep offline unit tests and minimal installs usable.
    trace = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment,misc]
    StatusCode = None  # type: ignore[assignment,misc]


def configure_tracing(
    *, service_name: str, enabled: bool, otlp_traces_endpoint: str = ""
) -> None:
    """Configure OTLP export, or safe console spans when no endpoint is configured."""

    if not enabled:
        return
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        if otlp_traces_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=otlp_traces_endpoint)
            exporter_name = "OTLP/HTTP"
        else:
            exporter = ConsoleSpanExporter()
            exporter_name = "console"
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry tracing enabled exporter=%s", exporter_name)
    except ImportError:
        logger.warning("OTEL_ENABLED=true but OpenTelemetry SDK/exporter is not installed")
    except Exception:
        logger.exception("Could not configure OpenTelemetry tracing")


@contextmanager
def trace_span(name: str, attributes: dict[str, object] | None = None) -> Iterator[object | None]:
    """Create a span and record errors without attaching prompts, transcripts, or audio."""

    if trace is None:
        yield None
        return
    tracer = trace.get_tracer("novatel.voice-agent")
    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            if value is not None and isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            if Status is not None and StatusCode is not None:
                span.set_status(Status(StatusCode.ERROR))
            raise

"""Service layer for the AI Creator Studio.

Everything that *does work* lives here: each service takes strongly typed input
and returns strongly typed domain objects. Services are stateless wherever
practical (holding only injected configuration), favour composition over
inheritance, and keep file I/O separate from core analysis logic so the logic
stays unit-testable.

Services arrive one per feature slice. Today:

* :mod:`backend.services.metadata` — gameplay video metadata extraction.
"""

"""Swap-seam contracts (``Protocol``\\ s) for interchangeable implementations.

This package is intentionally **empty for now**. We add an interface only when a
*second* implementation of something becomes real — e.g. once both an
``FFmpegEditor`` and a ``CapCutEditor`` exist, a ``VideoEditor`` Protocol here
lets the rest of the code depend on the contract instead of a concrete class.

Defining abstractions with a single implementation is premature, so nothing is
declared here yet. Services with exactly one implementation (such as
:class:`~backend.services.metadata.MetadataService`) are used directly.
"""

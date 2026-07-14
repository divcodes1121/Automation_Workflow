"""Asset sources — one adapter per provider, all behind :class:`AssetSource`.

Keyless sources download without credentials; keyed sources declare
``requires_key`` and read the key from the environment. Adding a source is a
new adapter here + one line in the manager's registry — no editor changes.
"""

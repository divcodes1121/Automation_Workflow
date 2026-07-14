"""Asset Manager — auto-downloads, organizes, and RESOLVES editing assets.

A standalone package (like ``analyzer/``): it never imports ``backend`` or the
editor, and the editor never references asset filenames. The editor asks for a
SYMBOLIC name (``emoji_fire``, ``impact_bass``, ``comic_boom``) and the manager
resolves it to a real file via the manifest — so asset packs can be added or
swapped with zero editor changes.

Assets come only from sources that offer an API or permit bulk download; each is
recorded with metadata (license included) in ``asset_manifest.json`` +
``asset_index.db``. Designed to scale to 10k+ assets.
"""

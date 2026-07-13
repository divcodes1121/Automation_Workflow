"""Independent feature modules for the studio pipeline.

Each future module owns exactly one concern and stays decoupled from the
others, so the :class:`~backend.workflow.WorkflowManager` can compose them in
any order. Planned modules:

* ``gameplay``  — highlight detection from raw footage.
* ``subtitles`` — transcription and subtitle generation.
* ``editor``    — FFmpeg-based long-form assembly.
* ``thumbnail`` — thumbnail image generation.
* ``youtube``   — YouTube upload/publish.
* ``shorts``    — vertical shorts rendering.
"""

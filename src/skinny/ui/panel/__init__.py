"""HoloViz Panel backend for the Skinny UI spec.

``backend.PanelTreeBuilder`` walks the same widget-tree
``skinny.ui.build_app_ui.build_main_ui`` produces and emits Panel widgets
+ a layout. Slot the result into the sidebar of ``web_app.create_panel_app``.
"""

import triton_viz


def launch_visualizer():
    from triton_viz.visualizer import draw

    serialize = draw.serialize_for_json

    def serialize_with_memoryview(obj):
        # Descriptor masks in this Triton Viz version can be NumPy memoryviews.
        if isinstance(obj, memoryview):
            return obj.tolist()
        return serialize(obj)

    draw.serialize_for_json = serialize_with_memoryview
    triton_viz.launch(share=False)

import threading
import time
import requests # Add this if you plan to use requests, otherwise remove
from app import gradio_app # Assuming app.py is in the root and contains gradio_app

def run_gradio_app():
    """Runs the Gradio app in a separate thread and returns a function to stop it."""
    
    # Find an available port
    port = 7860 # Default Gradio port
    # Add logic here if needed to find a free port dynamically, e.g. by trying to bind a socket.
    # For now, we'll assume 7860 is available or Gradio handles port conflicts.

    stop_event = threading.Event()

    def run():
        # To allow stopping, we might need to look into Gradio's app.stop() or similar,
        # or manage the server lifecycle more directly if possible.
        # The `launch` method blocks, so running it in a thread is correct.
        # `prevent_thread_lock=True` might be useful if issues arise, but Gradio's docs
        # should be checked for the best way to run/stop programmatically.
        # For now, we'll rely on the daemon thread property to exit when the main thread exits.
        # A more robust solution would involve an explicit stop mechanism.
        try:
            gradio_app.launch(server_port=port, prevent_thread_lock=True) # Added prevent_thread_lock
        except Exception as e:
            print(f"Error starting Gradio app: {e}")


    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    # Wait a bit for the server to start
    time.sleep(5) # Adjust as necessary

    print(f"Gradio app running on http://localhost:{port}")

    # How to stop the app:
    # Gradio's `Blocks.close()` or `App.close()` method should be used.
    # If `gradio_app` is an instance of `gradio.Blocks` or `gradio.App`.
    # Let's assume `gradio_app` has a `close` method.
    def stop_app():
        print("Attempting to stop Gradio app...")
        try:
            if hasattr(gradio_app, 'close') and callable(gradio_app.close):
                print("Calling gradio_app.close()...")
                gradio_app.close() # This is the standard way to close a Gradio app
                print("gradio_app.close() called.")
            else:
                # Fallback for older Gradio versions or different app structures
                # This part might be less reliable than app.close()
                if hasattr(gradio_app, 'server') and hasattr(gradio_app.server, 'close'):
                    print("Calling gradio_app.server.close()...")
                    gradio_app.server.close() # For underlying server if accessible
                    print("gradio_app.server.close() called.")
                
                # Attempt to shutdown the server via _server attribute (common in newer Gradio)
                # This is often an instance of uvicorn.Server
                if hasattr(gradio_app, '_server') and hasattr(gradio_app._server, 'shutdown')):
                    print("Calling gradio_app._server.shutdown()...")
                    # Uvicorn server needs to be shut down in its own thread if running in the main thread.
                    # However, since our app is in a daemon thread, this might be handled differently.
                    # For now, direct call, assuming it's safe or blocks until done.
                    # asyncio.run(gradio_app._server.shutdown()) # If it's an async shutdown
                    gradio_app._server.should_exit = True # Another way for uvicorn
                    # gradio_app._server.force_exit = True # More forceful
                    print("gradio_app._server.shutdown() related calls made.")


            # Ensure the thread itself is joined.
            # The stop_event isn't strictly necessary if app.close() is effective,
            # as it should cause launch() to return.
            if thread.is_alive():
                print("Gradio thread still alive, joining with timeout...")
                thread.join(timeout=10) # Increased timeout
                if thread.is_alive():
                    print("Warning: Gradio thread did not terminate after stop attempts and timeout.")
            else:
                print("Gradio thread has terminated.")
            
            print("Gradio app stop sequence complete.")

        except Exception as e:
            print(f"Exception during Gradio app stop: {e}")
        finally:
            # This event isn't used by launch() directly but can be for custom thread loops.
            # Keeping it for now in case the run() function evolves.
            if 'stop_event' in locals() and hasattr(stop_event, 'set'):
                stop_event.set()


    return f"http://localhost:{port}", stop_app

import pytest
from gradio_client import Client, file
import os
import numpy as np
from PIL import Image

# Assuming run_gradio_app is defined in this file as per previous step

@pytest.fixture(scope="module")
def gradio_server():
    app_url, stop_app_fn = run_gradio_app()
    yield app_url # Provide the URL to the tests
    stop_app_fn() # Stop the app after tests in this module are done

def test_image_inference(gradio_server):
    """Test inference with an image input."""
    client = Client(gradio_server)
    
    # Ensure the image path is correct relative to the project root
    # Assuming tests are run from the project root
    image_path = "ultralytics/assets/bus.jpg"
    if not os.path.exists(image_path):
        # Fallback if not found, perhaps due to different CWD
        # This path might need adjustment based on actual test execution context
        image_path = os.path.join(os.path.dirname(__file__), "..", "ultralytics/assets/bus.jpg")
        image_path = os.path.normpath(image_path)


    assert os.path.exists(image_path), f"Test image not found at {image_path}"

    # The API call through gr.Client needs to match how components are named/ordered in the Gradio app
    # We need to know the names or order of the input components for the target fn.
    # Let's inspect app.py's relevant click handler:
    # yolov12_infer.click(
    #     fn=run_inference,
    #     inputs=[image, video, model_id, image_size, conf_threshold, input_type],
    #     outputs=[output_image, output_video],
    # )
    # So, the inputs are: image, video, model_id, image_size, conf_threshold, input_type

    # For image inference:
    # image = path to image
    # video = None
    # model_id = e.g., "yolov12m.pt"
    # image_size = e.g., 640
    # conf_threshold = e.g., 0.25
    # input_type = "Image"

    job = client.predict(
        file(image_path),  # image input
        None,               # video input (None for image type)
        "yolov12m.pt",      # model_id
        640,                # image_size
        0.25,               # conf_threshold
        "Image",            # input_type
        api_name="/run_inference" # Specify the correct API endpoint name if needed.
                                  # Based on the click handler, Gradio might auto-generate this.
                                  # If a `gr.Button` has an `elem_id` or `api_name` is explicitly set, use that.
                                  # Otherwise, Gradio uses a default naming scheme, often related to the function name
                                  # or an index if multiple unnamed endpoints exist.
                                  # We might need to inspect the Gradio app's API info if this fails.
                                  # Often it's `/predict` or `/api/predict` or based on function name like `/run_inference`
    )
    
    # The `job.result()` or direct result depends on `gr.Client` version and how the endpoint is defined.
    # Assuming `predict` returns the direct output or a structure containing it.
    # The output for image inference is (annotated_image, None)
    output = job
    
    assert output is not None, "Prediction job did not return a result."
    
    # Check if the first element of the tuple (output_image) is the result
    # The output_image is expected to be a numpy array (from gr.Image(type="numpy"))
    # or a path to a file if `type` was different.
    # app.py: `output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)`
    # app.py: `return annotated_image[:, :, ::-1], None`
    # So, result[0] should be the numpy array.
    
    # The client receives a filepath to the output image by default
    # unless output component is gr.Image(type="numpy") AND client is configured for it.
    # By default, gr.Image sends image data as a base64 string or saves to temp file and sends path.
    # Let's assume it returns a filepath string.
    
    # The structure of `output` from client.predict for a function returning a tuple (image, video_path)
    # where image is a numpy array and video_path is None (for image input type) needs to be handled.
    # If `output_image` is `gr.Image(type="numpy")`, the client receives a filepath to a temporary image file.
    # Let's verify this behavior.
    
    # The `client.predict` method for an endpoint that has two outputs (output_image, output_video)
    # will return a tuple of results.
    # result[0] corresponds to output_image
    # result[1] corresponds to output_video
    
    assert isinstance(output, tuple), "Output should be a tuple (image_path, video_path_or_None)"
    image_output_path = output[0]
    video_output_result = output[1]

    assert image_output_path is not None, "Image output path should not be None"
    assert os.path.exists(image_output_path), f"Output image path {image_output_path} does not exist."

    # Try to open the image to verify it's a valid image file
    try:
        img = Image.open(image_output_path)
        img.verify() # Verify PIL can read it
        # Optionally, check dimensions or other properties if they are known/fixed
        assert img.format is not None, "Output image format could not be determined."
    except Exception as e:
        pytest.fail(f"Output image {image_output_path} is not a valid image: {e}")
    finally:
        # Clean up the temporary image file created by the Gradio client for the output
        if image_output_path and os.path.exists(image_output_path):
            os.remove(image_output_path)

    assert video_output_result is None, "Video output should be None for image inference."

import tempfile
import cv2 # OpenCV for creating a dummy video

def create_dummy_video(path, frames=30, width=640, height=480, fps=30):
    """Creates a simple dummy video file for testing."""
    fourcc = cv2.VideoWriter_fourcc(*'vp80') # Using VP8 for .webm, common for Gradio
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for _ in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8) # Black frame
        out.write(frame)
    out.release()
    return path

@pytest.fixture(scope="module")
def dummy_video_file():
    # Create a temporary video file
    # tempfile.NamedTemporaryFile can be tricky with reopening, so construct path manually
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "dummy_video.webm")
    
    created_video_path = create_dummy_video(video_path)
    
    yield created_video_path
    
    # Cleanup: remove the video file and directory
    if os.path.exists(created_video_path):
        os.remove(created_video_path)
    os.rmdir(temp_dir)


def test_video_inference(gradio_server, dummy_video_file):
    """Test inference with a video input."""
    client = Client(gradio_server)
    
    assert os.path.exists(dummy_video_file), f"Test video not found at {dummy_video_file}"

    # For video inference:
    # image = None
    # video = path to video
    # model_id = e.g., "yolov12m.pt"
    # image_size = e.g., 640 (used for frame processing)
    # conf_threshold = e.g., 0.25
    # input_type = "Video"

    job = client.predict(
        None,               # image input (None for video type)
        file(dummy_video_file), # video input
        "yolov12m.pt",      # model_id
        640,                # image_size
        0.25,               # conf_threshold
        "Video",            # input_type
        api_name="/run_inference" # Ensure this matches the API endpoint
    )
    
    output = job
    
    assert output is not None, "Prediction job did not return a result."
    
    # The output for video inference is (None, annotated_video_path)
    # result[0] corresponds to output_image (should be None)
    # result[1] corresponds to output_video (should be a filepath string)
    
    assert isinstance(output, tuple), "Output should be a tuple (image_path_or_None, video_path)"
    image_output_result = output[0]
    video_output_path = output[1]

    assert image_output_result is None, "Image output should be None for video inference."
    
    assert video_output_path is not None, "Video output path should not be None"
    assert os.path.exists(video_output_path), f"Output video path {video_output_path} does not exist."

    # Optionally, verify it's a valid video file (e.g., by trying to open with OpenCV)
    try:
        cap = cv2.VideoCapture(video_output_path)
        assert cap.isOpened(), "Could not open output video file."
        # Check some properties if known
        # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # assert width > 0 and height > 0, "Output video dimensions are invalid."
        cap.release()
    except Exception as e:
        pytest.fail(f"Output video {video_output_path} is not a valid video: {e}")
    finally:
        # Clean up the temporary video file created by the Gradio client for the output
        if video_output_path and os.path.exists(video_output_path):
            os.remove(video_output_path)

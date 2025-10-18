from django.shortcuts import render
import requests

def upload_view(request):
    """
    Handles uploading a photo and sending it to the FastAPI model.
    Renders 'upload.html' for GET requests.
    Renders 'result.html' for POST requests after sending the photo to FastAPI.
    """
    if request.method == "POST" and 'photo' in request.FILES:
        # Get the uploaded photo
        image = request.FILES['photo']

        # Prepare the file to send to FastAPI
        files = {'image': (image.name, image.read(), image.content_type)}

        # Try sending the image to FastAPI
        try:
            response = requests.post("http://127.0.0.1:8001/predict", files=files)
            response.raise_for_status()  # Raise an error for HTTP errors
            data = response.json()
        except requests.exceptions.RequestException as e:
            # Catch network errors, connection errors, or invalid responses
            data = {"error": f"FastAPI server not reachable: {e}"}

        # Render the result page
        return render(request, "result.html", {"result": data})

    # Render the upload page for GET requests or if no file is uploaded
    return render(request, "upload.html")

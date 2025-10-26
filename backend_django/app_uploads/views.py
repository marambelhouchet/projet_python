from django.shortcuts import render, redirect
import logging # Import logging module for server-side logs

# Import the agent runner function from your new agent file (rag_agent.py)
# Use a try-except block to handle potential import errors gracefully
try:
    from .rag_agent import run_agent
    agent_available = True
    logging.info("Successfully imported run_agent from .rag_agent")
except ImportError as e:
     # Log a critical error if the agent code cannot be imported
     logging.critical(f"CRITICAL: Could not import run_agent from .rag_agent: {e}", exc_info=True)
     logging.critical("CRITICAL: The main agent logic is missing or broken. The application will not function correctly.")
     agent_available = False
     run_agent = None # Define run_agent as None so checks later don't cause NameError
except Exception as e:
     # Catch any other unexpected errors during import
     logging.critical(f"CRITICAL: An unexpected error occurred during rag_agent import: {e}", exc_info=True)
     agent_available = False
     run_agent = None

# Get a logger instance for this module
logger = logging.getLogger(__name__)

def upload_view(request):
    """
    Handles the web request for the upload page.
    GET request: Shows the upload form.
    POST request: Gets user input, calls the agent, and shows the result.
    """
    # Handle POST request (when user submits the form)
    if request.method == 'POST':
        # Get the text question from the form's 'question' field
        user_question = request.POST.get('question', '').strip()
        logger.info(f"Received POST request. User question: '{user_question}'")

        # Get the uploaded image file (if any) from the form's 'photo' field
        image_file = request.FILES.get('photo', None)
        image_bytes = None # Initialize as None
        image_uploaded_successfully = False # Flag for the template

        # Default question if the user uploads an image but leaves the text box blank
        if not user_question and image_file:
             user_question = "Analyze the provided image and describe the findings, including potential diseases and treatments."
             logger.info(f"User submitted image but no question. Using default: '{user_question}'")
        # Handle case where user submits nothing at all
        elif not user_question and not image_file:
             # Redirect back to upload page with an error message perhaps?
             # For now, let's proceed but the agent will likely just respond based on this
             logger.warning("User submitted empty form. Using default question.")
             user_question = "Hello! Please provide a question or upload an image."


        # If an image file was uploaded, try to read its content (bytes)
        if image_file:
            try:
                image_bytes = image_file.read()
                # Basic validation: check if bytes were actually read
                if not image_bytes:
                    logger.warning(f"Uploaded image file '{image_file.name}' appears to be empty.")
                    image_bytes = None # Treat as if no image was uploaded
                else:
                    logger.info(f"Successfully read image file: '{image_file.name}', size: {len(image_bytes)} bytes")
                    image_uploaded_successfully = True # Set flag for template
            except Exception as e:
                # Log error if reading the file fails
                logger.error(f"Error reading uploaded image file '{image_file.name}': {e}", exc_info=True)
                image_bytes = None # Ensure image_bytes is None if reading failed

        # Check if the agent function was imported correctly before trying to call it
        if not agent_available or run_agent is None:
             agent_response = "ERROR: The Agri-Intel agent system is currently unavailable due to an internal setup issue. Please contact the administrator or check the server logs."
             logger.critical("Cannot execute agent: run_agent function is not available (import failed).")
        else:
            # Call the agent runner function with the user's question and image bytes (if any)
            try:
                logger.info(f"Dispatching request to agent. Image provided: {image_uploaded_successfully}")
                # This is where the agent logic in rag_agent.py takes over
                agent_response = run_agent(user_question, image_bytes)
                logger.info("Agent call finished. Received response.")
            except Exception as e:
                 # Catch unexpected errors during the agent's execution
                 logger.critical(f"!!! CRITICAL: Unhandled Error calling run_agent from view: {e} !!!", exc_info=True)
                 agent_response = f"An critical internal error occurred while processing your request. Please check the server logs for details. Error type: {type(e).__name__}"


        # Prepare the data to be sent to the result.html template
        context = {
            'question': user_question,         # The user's original (or default) question
            'agent_response': agent_response,  # The final answer generated by the agent
            'image_uploaded': image_uploaded_successfully # Boolean flag for the template
        }
        # Render the result page with the context data
        return render(request, 'result.html', context)

    # Handle GET request (when user first visits the page)
    else: # request.method == 'GET'
        logger.info("Received GET request for upload page.")
        # Just render the empty upload form
        return render(request, 'upload.html')
import os
import json
import requests
import base64
from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
import operator
import logging # Added for better logging

# --- Langchain specific imports ---
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, ToolMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

# --- Configuration ---
VECTORSTORE_PATH = "vectorstore" # Path relative to where Django runs (backend_django folder)
DISEASE_API_URL = "http://127.0.0.1:8001/predict" # Your FastAPI endpoint
LLM_MODEL = "llama3:latest" # Ollama model for agent decisions and final response
EMBEDDING_MODEL = "nomic-embed-text" # Ollama model for embeddings (must match create_vector_db.py)

# --- Setup Logging ---
# Configure logging to provide insights into the agent's operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Define Tools ---

# Tool 1: Disease Diagnostics (Calls your FastAPI model)
def run_disease_diagnostic(image_bytes: bytes) -> Dict[str, Any]:
    """
    Analyzes an image of a plant leaf to identify potential diseases.
    Args: image_bytes: The raw bytes of the image file.
    Returns: A dictionary containing the detected disease and confidence score, or an error message.
    """
    logger.info("--- Calling Disease Diagnostic Tool ---")
    if not image_bytes or len(image_bytes) < 100: # Basic check for empty/tiny image data
        logger.warning("Diagnostic Tool: Received empty or invalid image data.")
        return {"error": "No valid image provided to diagnostic tool."}
    try:
        # The key 'image' must match the parameter name in your FastAPI endpoint
        files = {'image': ("uploaded_image.jpg", image_bytes)}
        logger.info(f"Sending image ({len(image_bytes)} bytes) to {DISEASE_API_URL}")
        # Increased timeout for potentially slower model inference
        response = requests.post(DISEASE_API_URL, files=files, timeout=30)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        logger.info(f"Diagnostic Tool Result: {result}")
        # Ensure confidence is formatted correctly if present
        if 'confidence' in result and isinstance(result['confidence'], (int, float)):
             # Keep original float for potential logic, add formatted string for display/LLM context
             result['confidence_percent'] = f"{result['confidence']:.2f}%"
        return result
    except requests.exceptions.Timeout:
        error_msg = f"Failed to connect to diagnostic service: Connection timed out after 30 seconds."
        logger.error(f"ERROR in Diagnostic Tool: {error_msg}")
        return {"error": error_msg}
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to connect/communicate with diagnostic service at {DISEASE_API_URL}: {e}"
        logger.error(f"ERROR in Diagnostic Tool: {error_msg}")
        return {"error": error_msg}
    except json.JSONDecodeError as e:
         error_msg = f"Failed to decode JSON response from diagnostic service: {e}. Response text: {response.text[:200]}" # Log part of response
         logger.error(f"ERROR in Diagnostic Tool: {error_msg}")
         return {"error": "Diagnostic service returned invalid response."}
    except Exception as e:
        error_msg = f"Unexpected error during diagnosis: {e}"
        logger.error(f"ERROR in Diagnostic Tool: {error_msg}", exc_info=True) # Log full traceback
        return {"error": "An unexpected error occurred during image diagnosis."}

# Tool 2: RAG Retriever (Searches your vector database)
logger.info("Initializing RAG tool (loading vector store)...")
retriever_tool = None # Initialize as None in case of failure
try:
    # Check if the vectorstore directory exists where expected (relative to backend_django)
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.isdir(VECTORSTORE_PATH):
         raise FileNotFoundError(f"Vector store directory not found at '{VECTORSTORE_PATH}'. Please run 'create_vector_db.py' first.")

    logger.info(f"Loading FAISS vector store from: {VECTORSTORE_PATH}")
    # Initialize embeddings model (ensure Ollama is running)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    # Test connection
    logger.info("Testing embedding model connection...")
    _ = embeddings.embed_query("Test embedding.")
    logger.info("Embedding model connection successful.")

    # Load the local FAISS index
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True # Required for FAISS loading from local file
    )
    # Create a retriever object from the vector store
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 most relevant document chunks
    logger.info("FAISS vector store loaded successfully.")

    # Create the LangChain tool object that the agent can call
    retriever_tool = create_retriever_tool(
        retriever,
        "disease_info_retriever", # The name the LLM agent will use to call this tool
        "Searches and returns detailed information about specific plant diseases, including symptoms, treatments, and prevention methods. Use this tool when asked about a disease or how to manage it.",
    )
    logger.info("RAG retriever tool created successfully.")
except FileNotFoundError as e:
     logger.error(f"!!! RAG Tool INIT FAILED: {e} !!!")
     # Keep retriever_tool as None
except Exception as e:
    logger.error(f"\n!!! FATAL ERROR initializing RAG tool: {e} !!!", exc_info=True)
    logger.error("!!! Ensure 'vectorstore' folder exists in the 'backend_django' directory, Ollama is running with the embedding model '{EMBEDDING_MODEL}', and FAISS dependencies are correct. !!!")
    # Keep retriever_tool as None

# --- 2. Define Agent State ---
# This dictionary holds the information passed between steps in the graph
class AgentState(TypedDict):
    # List of messages (Human, AI, Tool) forming the conversation history
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Raw image bytes, passed only once at the beginning if provided
    image_bytes: Optional[bytes]
    # Store the result from the diagnostic tool separately
    diagnostic_result: Optional[Dict[str, Any]]

# --- 3. Define Agent Nodes and Edges ---

# Initialize the main LLM (Llama 3) for agent decisions and final response
try:
     logger.info(f"Initializing main agent LLM: {LLM_MODEL}")
     # Lower temperature for more deterministic, factual responses
     llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
     # Quick test to ensure connection
     # llm.invoke("Hi") # Can cause issues if Ollama isn't fully ready
     logger.info("Main agent LLM initialized.")
except Exception as e:
     logger.error(f"!!! FATAL ERROR initializing main LLM '{LLM_MODEL}': {e} !!!", exc_info=True)
     logger.error("!!! Make sure Ollama server is running and the model is downloaded (`ollama pull {LLM_MODEL}`). !!!")
     llm = None # Set to None if initialization failed

# List of tools the LLM agent can *choose* to call (only the RAG tool here)
# The image diagnostic tool is called explicitly in the preprocessing step.
tools = [retriever_tool] if retriever_tool else []
tool_executor = ToolExecutor(tools) # Executes the chosen tool (RAG search)

# Function defining the agent's decision logic (conditional edge)
def should_call_tools(state: AgentState) -> str:
    """Determines the next step after the agent LLM has responded."""
    if not state or not state.get('messages'):
        logger.warning("should_call_tools: Invalid state received, ending.")
        return "end" # End if state is invalid

    last_message = state['messages'][-1]

    # If the last message is not from the AI (e.g., Human input, Tool result),
    # the agent needs to process it.
    if not isinstance(last_message, AIMessage):
        logger.info("should_call_tools: Last message not AI, routing back to agent for processing.")
        return "continue_to_agent"

    # If the AI's last message contains instructions to call a tool:
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("should_call_tools: AI message requests tool call, routing to action.")
        return "continue_to_action"
    # If the AI's last message has no tool calls, it's the final answer.
    else:
        logger.info("should_call_tools: AI message has no tool calls, ending process.")
        return "end"

# Node that calls the main LLM agent
def call_agent_model(state: AgentState):
    """Invokes the agent LLM with the current message history and tools."""
    messages = state['messages']
    logger.info(f"\n--- Agent Node --- (Processing {len(messages)} messages)")
    # logger.debug(f"Agent History: {messages}") # Uncomment for very detailed logs

    if llm is None:
         logger.error("Agent Node: LLM not initialized!")
         # Return an error message directly in the correct format
         return {"messages": [AIMessage(content="Internal Error: Agent LLM is not available. Please check server logs.")]}

    # Invoke the LLM. It will look at the history and decide:
    # 1. Answer directly.
    # 2. Call the 'disease_info_retriever' tool.
    logger.info(f"Invoking LLM. Available tools: {[t.name for t in tools]}")
    response = llm.invoke(messages, tools=tools)
    logger.info(f"Agent Node LLM Response: Type={type(response)}, Contains Tool Calls={hasattr(response, 'tool_calls') and bool(response.tool_calls)}")
    # logger.debug(f"Agent Node LLM Raw Response: {response}") # Uncomment for detailed logs

    # The response (AIMessage) is appended to the state for the next step
    return {"messages": [response]}

# Node that executes the tool(s) called by the agent
def call_tool_executor(state: AgentState):
    """Executes the tool(s) requested by the agent's last message."""
    last_message = state['messages'][-1]
    logger.info("\n--- Action Node: Preparing Tool Execution ---")

    # Basic validation: Ensure the last message is an AIMessage with tool calls
    if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
         logger.warning(f"Action Node: Expected AI message with tool_calls, but found {type(last_message)}. Cannot execute tool.")
         # Return a ToolMessage indicating the error back to the agent
         return {"messages": [ToolMessage(content="Error: Invalid state for tool execution. Expected AIMessage with tool_calls.", tool_call_id="error-invalid-state")]}

    # Prepare tool invocations based on the AI's request
    tool_invocations = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id") # Get the ID for linking response
        if not tool_name or not tool_id:
             logger.warning(f"Action Node: Invalid tool call structure received: {tool_call}")
             continue # Skip invalid calls

        action = ToolInvocation(
            tool=tool_name,
            tool_input=tool_args,
            tool_call_id=tool_id
        )
        tool_invocations.append(action)
        logger.info(f"Action Node: Prepared call for tool '{tool_name}' (ID: {tool_id}) with args: {tool_args}")

    if not tool_invocations:
         logger.warning("Action Node: No valid tool invocations prepared.")
         return {"messages": [ToolMessage(content="Error: No valid tools were called.", tool_call_id="error-no-valid-calls")]}


    # Execute the tool(s) using Langchain's ToolExecutor
    # It runs the function associated with the tool name (e.g., retriever_tool.func)
    responses = tool_executor.batch(tool_invocations, return_exceptions=True)
    logger.info(f"Action Node: Raw Tool Execution Responses: {responses}")

    # Format the results as ToolMessage objects for the agent to process
    tool_messages = []
    for i, response in enumerate(responses):
         # Check if the tool execution resulted in an error
         is_error = isinstance(response, Exception)
         # Provide informative content string, whether success or error
         content_str = f"Error executing tool '{tool_invocations[i].tool}': {response}" if is_error else str(response)

         tool_messages.append(ToolMessage(
             content=content_str,
             tool_call_id=tool_invocations[i].tool_call_id # Link response back to the specific call
         ))
         if is_error:
              logger.error(f"Action Node: Error response from tool '{tool_invocations[i].tool}' (ID: {tool_invocations[i].tool_call_id}): {response}")
         else:
              # Log success, maybe truncate long RAG results for clarity
              log_content = content_str[:200] + "..." if len(content_str) > 200 else content_str
              logger.info(f"Action Node: Successful response from tool '{tool_invocations[i].tool}' (ID: {tool_invocations[i].tool_call_id}): {log_content}")

    # Return the tool results to be added to the agent state
    return {"messages": tool_messages}

# --- 4. Define Pre-processing Node for Initial Input Handling ---
def preprocess_input(state: AgentState):
    """
    Handles the initial user input. If an image is present, it calls the
    disease diagnostic tool. It then constructs a detailed initial prompt
    for the main agent LLM based on the user query and any diagnostic results.
    """
    logger.info("\n--- Preprocessing Node ---")
    # Extract the original user question (it's the first message)
    user_question = state['messages'][0].content
    image_bytes = state.get('image_bytes')
    diagnostic_result_data = None # Variable to store the outcome of the diagnosis

    # If image data exists, call the diagnostic tool function
    if image_bytes:
        logger.info("Preprocessing: Image data found, calling diagnostic tool...")
        diagnostic_result_data = run_disease_diagnostic(image_bytes)
    else:
        logger.info("Preprocessing: No image data provided.")

    # Construct the detailed prompt for the first call to the main agent LLM
    # This prompt guides the agent on what information it has and what it should do next
    initial_prompt_parts = [f"User query: '{user_question}'"] # Start with the user's raw query

    if diagnostic_result_data:
        # Append context based on diagnostic outcome
        if "error" in diagnostic_result_data:
            initial_prompt_parts.append(f"\n\nContext: Image analysis was attempted but FAILED with error: {diagnostic_result_data['error']}. Please address the user's text query only and inform the user about the image analysis failure if relevant.")
        elif "disease" in diagnostic_result_data:
            disease = diagnostic_result_data['disease']
            # Use confidence_percent if available, otherwise format the raw confidence
            conf_str = diagnostic_result_data.get('confidence_percent',
                                                f"{diagnostic_result_data.get('confidence', 0)*100:.0f}%" if 'confidence' in diagnostic_result_data else "N/A")
            initial_prompt_parts.append(f"\n\nContext: An image was provided and analyzed. The identified disease is likely **{disease}** (Confidence: {conf_str}).")

            # Give explicit instruction to use the RAG tool for the detected disease
            if retriever_tool: # Only instruct if the tool is actually available
                 initial_prompt_parts.append(f"\n\nInstruction: Based on this diagnosis, use the '{retriever_tool.name}' tool to search for specific information (symptoms, treatment, prevention) about '{disease}'. Then, synthesize this information to answer the user's original query: '{user_question}'.")
            else:
                 initial_prompt_parts.append("\n\nInstruction: Provide advice based on the detected disease and the user's query using your general knowledge, as the specific information retrieval tool is unavailable.")
        else:
             # Handle unexpected but non-error response from diagnostic tool
             initial_prompt_parts.append("\n\nContext: Image analysis completed but returned an unexpected result (no disease or error). Please address the user's text query only.")
    else:
        # No image was provided
        initial_prompt_parts.append("\n\nContext: No image was provided.")
        # Give instruction to use RAG if appropriate for the text query
        if retriever_tool:
            initial_prompt_parts.append(f"\n\nInstruction: Answer the user's query. If the query asks about a specific disease, symptoms, treatment, or requires detailed agricultural knowledge, use the '{retriever_tool.name}' tool to find relevant information. Otherwise, answer using your general knowledge.")
        else:
            initial_prompt_parts.append("\n\nInstruction: Answer the user's query using your general knowledge, as the specific information retrieval tool is unavailable.")

    # Combine the parts into the final prompt
    initial_agent_prompt = "".join(initial_prompt_parts)

    logger.info(f"Preprocessing: Generated initial agent prompt (length {len(initial_agent_prompt)} chars):\n{initial_agent_prompt[:500]}...") # Log start of prompt

    # Return the new state:
    # - messages: Replace the original simple HumanMessage with this detailed one.
    # - diagnostic_result: Store the raw diagnostic result separately in the state.
    return {
        "messages": [HumanMessage(content=initial_agent_prompt)],
        "diagnostic_result": diagnostic_result_data
        }

# --- 5. Build the LangGraph Graph ---
logger.info("Building agent graph workflow...")
workflow = StateGraph(AgentState)

# Add the defined functions as nodes in the graph
workflow.add_node("preprocess", preprocess_input) # Step 1: Handle input & diagnostics
workflow.add_node("agent", call_agent_model)       # Step 2 & 4: LLM decides/responds
workflow.add_node("action", call_tool_executor)    # Step 3: Execute RAG tool if needed

# Define the connections (edges) between the nodes
workflow.set_entry_point("preprocess")      # The graph starts at the preprocess node
workflow.add_edge("preprocess", "agent")    # After preprocessing, always go to the agent

# Add the conditional logic: after the agent runs, decide where to go next
workflow.add_conditional_edges(
    "agent",                            # The source node is the agent
    should_call_tools,                  # The function that makes the decision
    {
        "continue_to_action": "action", # If agent wants to use a tool, go to action node
        "continue_to_agent": "agent",   # If a tool just ran, go back to agent to process result (THIS CASE SHOULDN'T HAPPEN WITH CURRENT should_call_tools LOGIC, but included for robustness)
        "end": END                      # If agent gives final answer, end the graph execution
    }
)

# After a tool is executed (action node), the results must go back to the agent node
workflow.add_edge("action", "agent")

# Compile the graph definition into a runnable application
try:
     agent_graph = workflow.compile()
     logger.info("Agent graph compiled successfully.")
except Exception as e:
     logger.error(f"!!! FATAL ERROR compiling agent graph: {e} !!!", exc_info=True)
     agent_graph = None # Set to None on failure

# --- 6. Function to Run the Agent ---
def run_agent(user_input: str, image_bytes_param: Optional[bytes] = None) -> str:
    """
    Executes the compiled agent graph with the user's query and optional image bytes.
    Returns the final response string from the agent.
    """
    if agent_graph is None:
         logger.error("run_agent: Attempted to run but agent graph is not compiled.")
         return "Error: Agent graph failed to compile. Check server logs."
    if llm is None:
         logger.error("run_agent: Attempted to run but main LLM is not initialized.")
         return "Error: Agent LLM failed to initialize. Check server logs."
    if not tools: # Check if the tools list (specifically retriever_tool) is empty
         logger.warning("run_agent: RAG tool is not available (failed to load vector store?). Agent will rely on image diagnostics and general knowledge.")
         # Allow execution, but RAG won't work as expected

    # Prepare the initial state to feed into the graph
    initial_state = AgentState(
        messages=[HumanMessage(content=user_input)], # Start with just the user's raw query
        image_bytes=image_bytes_param,               # Pass image bytes if provided
        diagnostic_result=None                       # Initialize diagnostic result as None
    )

    logger.info(f"\n--- Running Agent --- Query: '{user_input}', Image Provided: {image_bytes_param is not None} ---")
    final_response_content = "Agent process completed, but no final response was generated. Please check logs." # Default/fallback message

    try:
        # Execute the graph stream. This provides visibility into each step's output.
        events = agent_graph.stream(initial_state, stream_mode="values")
        final_state = None
        logger.info("Streaming agent execution events...")
        for i, event in enumerate(events):
            logger.info(f"Graph Step {i+1}: Node output received.")
            # You could inspect 'event' here for debugging if needed
            # logger.debug(f"Event State: {event}")
            final_state = event # Keep track of the latest state after each node runs

        logger.info("Agent execution stream finished.")

        # After the stream finishes, extract the actual final response
        # The final response should be the 'content' of the last AIMessage in the 'messages' list,
        # provided it doesn't contain tool calls.
        if final_state and final_state.get('messages'):
            last_message = final_state['messages'][-1]
            logger.info(f"Final state's last message type: {type(last_message)}")
            if isinstance(last_message, AIMessage) and not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                final_response_content = last_message.content
                logger.info("run_agent: Extracted final response from the last AI message in final state.")
            else:
                # Fallback: If the last message isn't the final answer (e.g., graph ended unexpectedly),
                # search backwards through the message history for the last valid AI response.
                logger.warning("run_agent: Last message in final state was not a simple AI response. Searching history...")
                for msg in reversed(final_state['messages']):
                    if isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                        final_response_content = msg.content
                        logger.info(f"run_agent: Found last valid AI response at index {-final_state['messages'].index(msg)-1} in history.")
                        break
                else:
                     logger.error("run_agent: Could not find any valid final AI response in the message history.")

        else:
             logger.error("run_agent: Agent finished, but the final state or messages list was empty or invalid.")


        logger.info(f"--- Agent Run Complete --- Final Response (first 200 chars): {final_response_content[:200]}... ---")
        return final_response_content

    except Exception as e:
        logger.error(f"!!! CRITICAL ERROR during agent graph execution: {e} !!!", exc_info=True)
        return f"An unexpected error occurred while processing your request. Please check the server logs for details."

# --- Standalone Test Section ---
if __name__ == "__main__":
    # This block runs only if the script is executed directly (e.g., python rag_agent.py)
    # Useful for testing the agent logic without the full Django setup.
    print("\n--- Running Standalone Agent Test ---")
    # Ensure prerequisites are met (Ollama running, vectorstore exists) before running this

    # Test 1: Text only query - should use RAG
    print("\n--- Test 1: Text Query (RAG expected) ---")
    response1 = run_agent("What is Apple Scab and how is it treated?")
    print("\nFinal Response (Test 1):\n", response1)
    print("-" * 50)

    # Test 2: Text + Fake Image (Diagnostic call should fail, agent should note failure and try RAG/general knowledge)
    print("\n--- Test 2: Text + Fake Image Query (Diagnostic Fail expected) ---")
    fake_image_bytes = b"this is definitely not image data"
    response2 = run_agent("What is wrong with my apple tree leaf?", fake_image_bytes)
    print("\nFinal Response (Test 2):\n", response2)
    print("-" * 50)

    # Test 3: Text + Real Image (Requires a valid image file named 'test_leaf.jpg' in the same directory)
    print("\n--- Test 3: Text + Real Image Query (Diagnostic + RAG expected) ---")
    TEST_IMAGE_PATH = "test_leaf.jpg" # Make sure you have a test image with this name
    try:
        with open(TEST_IMAGE_PATH, "rb") as f:
            real_image_bytes = f.read()
        print(f"   (Using image: {TEST_IMAGE_PATH})")
        response3 = run_agent("What is this disease and how do I prevent it in the future?", real_image_bytes)
        print("\nFinal Response (Test 3):\n", response3)
    except FileNotFoundError:
        print(f"   Skipping real image test: Test image '{TEST_IMAGE_PATH}' not found in the current directory ({os.getcwd()}).")
    except Exception as e:
         print(f"   Error during real image test: {e}")

    print("\n--- Standalone Test Complete ---")
    


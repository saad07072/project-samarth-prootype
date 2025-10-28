import pandas as pd
import warnings
import re
import json
import requests
import time  # For retry logic
from flask import Flask, request, jsonify, render_template_string

# --- Configuration ---
AGRI_DATA_PATH = 'agri.csv'
RAIN_DATA_PATH = 'rain.csv'
SOIL_DATA_PATH = 'soil.csv'

# Gemini API Configuration
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! IMPORTANT: YOU MUST ADD YOUR OWN API KEY HERE FOR THE APP TO WORK      !!!
# !!! Get a key from Google AI Studio: https://aistudio.google.com/app/keys  !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE" # <--- PASTE YOUR KEY HERE

# NOTE: If API_KEY is set to "PASTE_YOUR_GEMINI_API_KEY_HERE" or is empty, the app will not work.
MODEL_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# Global variable to hold our master DataFrame
master_df = None
df_columns = "[]"

app = Flask(__name__)

# --- Phase 1: Data Discovery & Integration ---

def load_and_prepare_data():
    """
    Loads, cleans, aggregates, and merges the three disparate data sources
    into a single, analysis-ready DataFrame.
    
    This function is run once when the server starts.
    """
    global master_df, df_columns
    print("Starting data integration...")
    
    # Suppress warnings for cleaner output
    warnings.simplefilter(action='ignore', category=FutureWarning)

    try:
        # 1. Load Agricultural Data (Annual)
        df_agri = pd.read_csv(AGRI_DATA_PATH, on_bad_lines='skip')
        df_agri.rename(columns={'State Name': 'State', 'Dist Name': 'District'}, inplace=True)
        
        # 2. Load and Aggregate Rainfall Data (Daily)
        df_rain = pd.read_csv(RAIN_DATA_PATH, on_bad_lines='skip')
        # Convert Date to datetime
        df_rain['Date'] = pd.to_datetime(df_rain['Date'], errors='coerce')
        # Extract 'Year' from 'Date' for robust grouping, fallback to 'Year' column
        df_rain['Agg_Year'] = df_rain['Date'].dt.year.fillna(df_rain['Year']).astype(int)
        
        df_rain_annual = df_rain.groupby(['Agg_Year', 'State', 'District'])['Avg_rainfall'].sum().reset_index()
        df_rain_annual.rename(columns={'Avg_rainfall': 'Total_Annual_Rainfall_mm', 'Agg_Year': 'Year'}, inplace=True)

        # 3. Load and Aggregate Soil Moisture Data (Daily)
        df_soil = pd.read_csv(SOIL_DATA_PATH, on_bad_lines='skip')
        # Convert Date to datetime
        df_soil['Date'] = pd.to_datetime(df_soil['Date'], errors='coerce')
        # Extract 'Year' from 'Date' for robust grouping, fallback to 'Year' column
        df_soil['Agg_Year'] = df_soil['Date'].dt.year.fillna(df_soil['Year']).astype(int)
        
        df_soil_annual = df_soil.groupby(['Agg_Year', 'State', 'District'])['Avg_smlvl_at15cm'].mean().reset_index()
        df_soil_annual.rename(columns={'Avg_smlvl_at15cm': 'Mean_Annual_Soil_Moisture', 'Agg_Year': 'Year'}, inplace=True)

        # 4. Standardize all keys for merging
        for df in [df_agri, df_rain_annual, df_soil_annual]:
            if 'State' in df.columns:
                df['State'] = df['State'].str.strip().str.title()
            if 'District' in df.columns:
                df['District'] = df['District'].str.strip().str.title()
            if 'Year' in df.columns:
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce').dropna().astype(int)
                
        # Drop rows with NaT/NaN years which are unusable
        df_agri.dropna(subset=['Year'], inplace=True)
        df_rain_annual.dropna(subset=['Year'], inplace=True)
        df_soil_annual.dropna(subset=['Year'], inplace=True)

        # 5. Merge the DataFrames
        print("Merging DataFrames...")
        df_merged = pd.merge(df_agri, df_rain_annual, on=['Year', 'State', 'District'], how='left')
        master_df = pd.merge(df_merged, df_soil_annual, on=['Year', 'State', 'District'], how='left')
        
        # Store column names as a JSON string for the LLM prompt
        df_columns = json.dumps(master_df.columns.tolist())
        
        print(f"Data integration complete. Master DataFrame has {len(master_df)} rows.")
        print("Sample of merged data:")
        print(master_df[['Year', 'State', 'District', 'RICE PRODUCTION (1000 tons)', 'Total_Annual_Rainfall_mm', 'Mean_Annual_Soil_Moisture']].dropna().head())
        print(f"Columns available for queries: {df_columns}")

    except FileNotFoundError as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: Data file not found: {e.filename}")
        print(f"Please make sure '{AGRI_DATA_PATH}', '{RAIN_DATA_PATH}', and '{SOIL_DATA_PATH}' are in the same directory as app.py")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        master_df = None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        master_df = None

# --- Phase 2: The Intelligent Q&A System ---

def query_model(system_config, user_query):
    """
    Calls the generative model API with retry logic.
    """
    print(f"Querying model. Query: {user_query[:50]}...")
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_config}]
        },
        "contents": [{
            "parts": [{"text": user_query}]
        }]
    }
    
    retries = 3
    base_delay = 1 # seconds
    
    for attempt in range(retries):
        try:
            response = requests.post(MODEL_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=120)

            # Check for specific server errors that are safe to retry
            if response.status_code in [500, 502, 503, 504]:
                print(f"Warning: Received HTTP {response.status_code} (Service Unavailable). Retrying in {base_delay}s...")
                time.sleep(base_delay)
                base_delay *= 2 # Exponential backoff
                continue # Go to the next attempt

            # Raise an exception for other bad status codes (like 400, 403, 404)
            response.raise_for_status() 
            
            data = response.json()
            
            # Navigate the response structure to get the text
            if (data.get('candidates') and 
                data['candidates'][0].get('content') and 
                data['candidates'][0]['content'].get('parts')):
                
                text_response = data['candidates'][0]['content']['parts'][0]['text']
                print("Gemini API call successful.")
                return text_response, None # Success, exit the function
            else:
                print(f"Error: Unexpected API response format: {data}")
                return None, f"Error: Unexpected API response format: {data.get('error', 'Unknown')}"

        except requests.exceptions.RequestException as e:
            print(f"Error calling model API (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay)
                base_delay *= 2 # Exponential backoff
            else:
                return None, f"Error: Could not connect to model API after {retries} attempts. {e}" # All retries failed
        except Exception as e:
            print(f"An unexpected error occurred during API call: {e}")
            return None, f"An unexpected error occurred: {e}"

    # This line is reached if all retries fail with 50x errors
    return None, f"Error: Service unavailable after {retries} attempts."


def clean_generated_code(raw_code):
    """
    Cleans the raw output from the LLM to extract only the Python code.
    """
    # Remove markdown code block delimiters
    if raw_code.startswith("```python"):
        raw_code = raw_code[9:]
    if raw_code.endswith("```"):
        raw_code = raw_code[:-3]
    return raw_code.strip()

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    The main Q&A endpoint.
    Receives a question, generates code, executes it, and synthesizes an answer.
    """
    # === FIX 1: Removed redundant local import. The global import at line 1 is used. ===
    
    if API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE" or API_KEY == "":
        print("ERROR: API_KEY is not set in app.py")
        return jsonify({"error": "Server-side configuration error: The Gemini API key is not set. Please add your API key to the 'API_KEY' variable in app.py."}), 500
        
    if master_df is None:
        return jsonify({"error": "Data is not loaded. Please check server logs for file path errors."}), 500

    # === FIX 2: Replaced unsafe `request.json.get` with robust `request.get_json()` ===
    data = request.get_json()
    if not data:
        print("Error: Request received without a valid JSON body or Content-Type header.")
        return jsonify({"error": "Invalid request: No JSON body or incorrect Content-Type."}), 400
        
    question = data.get('question')
    if not question:
        print("Error: JSON body received, but 'question' field is missing.")
        return jsonify({"error": "Invalid request: 'question' field is missing from JSON body."}), 400

    print(f"\nReceived new question: {question}")

    # Step 1: Generate Code
    CODE_GEN_CONFIG = f"""
TASK: Write Python code to answer a user's question using a pandas DataFrame.
CONTEXT: The DataFrame is named `df`.
COLUMNS: {df_columns}

RULES:
1.  Output MUST be a single, standalone Python code block.
2.  The last line of code MUST be `result = ...` (assigning the final answer).
3.  Do NOT use `print()`.
4.  Only use the `pandas` library (aliased as `pd`) and the `df` variable.
5.  Perform case-insensitive string matching (e.g., `df['State'].str.title() == 'Maharashtra'`).
6.  Do NOT use `os`, `subprocess`, `eval`, `exec`, or `requests`.

EXAMPLE_QUESTION: "What was the total RICE PRODUCTION in Maharashtra in 2010?"
EXAMPLE_CODE:
df_filtered = df[
    (df['State'].str.title() == 'Maharashtra') &
    (df['Year'] == 2010)
]
result = df_filtered['RICE PRODUCTION (1000 tons)'].sum()
"""

    generated_code, err = query_model(CODE_GEN_CONFIG, question)
    if err:
        return jsonify({"error": err}), 500

    cleaned_code = clean_generated_code(generated_code)
    print(f"--- Generated Code ---\n{cleaned_code}\n------------------------")

    # --- Step 2: Execute Code (Safely!) ---
    data_result = None
    exec_error = None
    
    # Create a local context for exec. Pass a copy of the dataframe.
    # 'pd' now refers to the global import from the top of the file.
    local_context = {"df": master_df.copy(), "pd": pd}
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!! WARNING FOR PRODUCTION: exec() is a major security risk. !!!
    # !!! This is acceptable ONLY for a contained prototype demo.  !!!
    # !!! A production system MUST use a secure sandboxed env.     !!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    try:
        # Pass an empty dict {} for globals for a stricter sandbox
        exec(cleaned_code, {}, local_context)
        data_result = local_context.get('result')
        
        # Convert pandas objects to JSON-serializable formats
        if isinstance(data_result, pd.DataFrame):
            data_result = data_result.to_json(orient='records')
        elif isinstance(data_result, pd.Series):
            data_result = data_result.to_json(orient='records')
        elif isinstance(data_result, (float, int, str, list, dict)):
            pass # Already serializable
        else:
            data_result = str(data_result) # Fallback
            
        print(f"Code execution successful. Result: {str(data_result)[:150]}...")

    except Exception as e:
        print(f"Error executing generated code: {e}")
        exec_error = str(e)

    # --- Step 3: Synthesize Answer (LLM Call 2) ---
    if exec_error:
        # If code execution failed, ask the LLM to explain the error
        ERROR_CONFIG = f"""
TASK: Explain a Python code execution error to a user in simple terms.
USER_QUESTION: "{question}"
FAILED_CODE:
{cleaned_code}
ERROR_MESSAGE: "{exec_error}"

Provide a user-friendly explanation of what went wrong and suggest how to rephrase the question.
"""
        final_answer, err = query_model(ERROR_CONFIG, question)
    
    else:
        # If code execution succeeded, synthesize the answer
        SYNTHESIS_CONFIG = f"""
TASK: Provide a clear, natural language answer to a user's question based *only* on the provided data.
CONTEXT: You are an analyst for "Project Samarth".
USER_QUESTION: "{question}"
DATA_RESULT:
{data_result}

RULES:
1.  Formulate a direct answer using *only* the data in DATA_RESULT.
2.  If DATA_RESULT is empty or null, state that the data is not available.
3.  End the response with the following mandatory citation:
    "[Sources: {AGRI_DATA_PATH}, {RAIN_DATA_PATH}, {SOIL_DATA_PATH}]"
"""
        final_answer, err = query_model(SYNTHESIS_CONFIG, question)

    if err:
        return jsonify({"error": err}), 500
        
    # --- Step 4: Return JSON Response ---
    response_data = {
        "answer": final_answer,
        "data_result": data_result,
        "generated_code": cleaned_code,
        "error": exec_error
    }
    
    return jsonify(response_data)


# --- Front-End: HTML/CSS/JS ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="h-full bg-gray-900">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Samarth</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .spinner {
            border-top-color: #3498db;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="h-full flex flex-col items-center justify-start text-gray-100 p-4 md:p-8">
    <div class="w-full max-w-4xl">
        <!-- Header -->
        <header class="text-center mb-8">
            <h1 class="text-4xl md:text-5xl font-bold text-white mb-2">Project Samarth</h1>
            <p class="text-lg md:text-xl text-blue-300">Intelligent Q&A for India's Agricultural & Climate Data</p>
        </header>

        <!-- Main Content -->
        <main class="w-full">
            <!-- Input Form -->
            <form id="qa-form" class="mb-8">
                <div class="relative">
                    <input type="text" id="question-input"
                        class="w-full p-4 pr-20 bg-gray-800 border-2 border-gray-700 rounded-lg text-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="e.g., What was the total RICE PRODUCTION in Maharashtra in 2010?" required>
                    <button type="submit" id="ask-button"
                        class="absolute top-2 right-2 px-6 py-2 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:bg-gray-500">
                        Ask
                    </button>
                </div>
            </form>

            <!-- Loading Spinner -->
            <div id="loader" class="hidden flex-col items-center justify-center my-10">
                <div class="spinner w-12 h-12 rounded-full border-4 border-gray-700 border-t-blue-500 mb-4"></div>
                <p class="text-lg text-gray-400">Synthesizing answer... This may take a moment.</p>
            </div>

            <!-- Results Area -->
            <div id="results-container" class="hidden bg-gray-800 rounded-lg shadow-2xl overflow-hidden">
                <!-- Synthesized Answer -->
                <div class="p-6 border-b border-gray-700">
                    <h2 class="text-2xl font-semibold text-white mb-4">Synthesized Answer</h2>
                    <div id="answer-text" class="text-lg text-gray-200 leading-relaxed whitespace-pre-wrap"></div>
                </div>

                <!-- Traceability Details -->
                <div class="p-6">
                    <h3 class="text-xl font-semibold text-blue-300 mb-4">Traceability & Data</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <!-- Generated Code -->
                        <div>
                            <label class="block text-sm font-medium text-gray-400 mb-2">Generated Code</label>
                            <pre class="bg-gray-900 rounded-lg p-4 text-sm text-yellow-300 overflow-x-auto">
<code id="generated-code"></code>
                            </pre>
                        </div>
                        <!-- Raw Data Result -->
                        <div>
                            <label class="block text-sm font-medium text-gray-400 mb-2">Raw Data Result</label>
                            <pre class="bg-gray-900 rounded-lg p-4 text-sm text-gray-300 overflow-auto max-h-60">
<code id="data-result"></code>
                            </pre>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Error Message -->
            <div id="error-container" class="hidden bg-red-900 border border-red-700 text-red-100 p-4 rounded-lg">
                <h3 class="font-bold mb-2">An Error Occurred</h3>
                <p id="error-text"></p>
            </div>

            <!-- Sample Questions -->
            <div class="mt-12">
                <h3 class="text-xl font-semibold text-gray-300 mb-4">Sample Questions to Try</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <button class="sample-q-btn">What was the total RICE PRODUCTION in Maharashtra in 2010?</button>
                    <button class="sample-q-btn">Which 5 districts in 'Andhra Pradesh' had the highest 'Total_Annual_Rainfall_mm' in 2012?</button>
                    <button class="sample-q-btn">What is the average 'Mean_Annual_Soil_Moisture' in the state of 'Gujarat' for the year 2011?</button>
                    <button class="sample-q-btn">Compare the total 'WHEAT PRODUCTION (1000 tons)' in 'Punjab' and 'Haryana' for 2014.</button>
                </div>
            </div>
        </main>
    </div>

    <script>
        const form = document.getElementById('qa-form');
        const input = document.getElementById('question-input');
        const askButton = document.getElementById('ask-button');
        const loader = document.getElementById('loader');
        const resultsContainer = document.getElementById('results-container');
        const errorContainer = document.getElementById('error-container');
        
        const answerText = document.getElementById('answer-text');
        const generatedCode = document.getElementById('generated-code');
        const dataResult = document.getElementById('data-result');
        const errorText = document.getElementById('error-text');

        // Style sample question buttons
        document.querySelectorAll('.sample-q-btn').forEach(button => {
            button.className = "p-3 bg-gray-800 border border-gray-700 rounded-lg text-left text-gray-300 hover:bg-gray-700 hover:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500";
            button.addEventListener('click', () => {
                input.value = button.textContent;
                form.dispatchEvent(new Event('submit'));
            });
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = input.value.trim();
            if (!question) return;

            // --- Reset UI ---
            askButton.disabled = true;
            askButton.textContent = "Asking...";
            loader.classList.remove('hidden');
            resultsContainer.classList.add('hidden');
            errorContainer.classList.add('hidden');

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || `HTTP error! status: ${response.status}`);
                }

                // --- Populate Results ---
                answerText.textContent = data.answer;
                generatedCode.textContent = data.generated_code;
                
                // Try to pretty-print JSON data
                try {
                    // Check if data_result is already a JSON string
                    const jsonData = JSON.parse(data.data_result);
                    dataResult.textContent = JSON.stringify(jsonData, null, 2);
                } catch {
                    // If it's not a JSON string, just display it
                    dataResult.textContent = data.data_result;
                }

                if(data.error) {
                    // Show a non-fatal error (e.g., code execution error)
                    showError(`Code Execution Error: ${data.error}\\n\\nSee generated code for details.`);
                } else {
                    errorContainer.classList.add('hidden');
                }
                
                resultsContainer.classList.remove('hidden');

            } catch (err) {
                console.error('Fetch error:', err);
                showError(err.message);
            } finally {
                // --- Restore UI ---
                loader.classList.add('hidden');
                askButton.disabled = false;
                askButton.textContent = "Ask";
            }
        });
        
        function showError(message) {
            errorText.textContent = message;
            errorContainer.classList.remove('hidden');
            resultsContainer.classList.add('hidden');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template_string(HTML_TEMPLATE)

# --- Main execution ---
if __name__ == "__main__":
    # Load and prepare data *before* starting the app
    load_and_prepare_data()
    
    if master_df is not None:
        print("\n--- Project Samarth Server is RUNNING ---")
        print("Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.")
        print("------------------------------------------\n")
        app.run(debug=True, port=5000)
    else:
        print("\n--- Project Samarth Server FAILED TO START ---")
        print("Master DataFrame could not be loaded. Please check file paths and errors above.")
        print("--------------------------------------------\n")


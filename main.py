import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

# Set up the API key
genai.configure(api_key='AIzaSyDXo5O9YZJSt-QQh_a6qJD0Jq7QJADzCHE')  # Replace with your actual API key

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the input model
class Notes(BaseModel):
    text: str

# Initialize the model
model = genai.GenerativeModel('gemini-pro')

# Load extraction categories from config file
with open('extraction_config.json', 'r') as config_file:
    config = json.load(config_file)
    extraction_categories = config['extraction_categories']

@app.post("/process_notes")
async def process_notes(notes: Notes):
    category_prompt = "\n".join([f"- {category}" for category in extraction_categories])
    
    prompt = f"""
    You are a BPO assistant. Summarize the following notes and extract key details for these categories:
    {category_prompt}

    The output must be valid JSON and strictly follow this format:
    {{
        "summary": "A brief summary of the conversation",
        "details": {{
            "Category 1": "Extracted information for Category 1",
            "Category 2": "Extracted information for Category 2"
        }}
    }}

    Ensure all keys and values are properly quoted. Do not include any text outside of this JSON structure.

    Notes: {notes.text}
    """

    try:
        response = model.generate_content(prompt)  # Remove temperature parameter
        
        # Attempt to parse the response as JSON
        output = json.loads(response.text)
        
        # Validate that the output matches our expected structure
        if "summary" not in output or "details" not in output:
            raise ValueError("Response missing required fields")
        
        return output
    except json.JSONDecodeError as e:
        return {"error": f"Failed to generate valid JSON: {str(e)}", "raw_response": response.text if 'response' in locals() else "No response generated"}
    except ValueError as e:
        return {"error": str(e), "raw_response": response.text if 'response' in locals() else "No response generated"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@app.get("/extraction_categories")
async def get_extraction_categories():
    return {"categories": extraction_categories}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

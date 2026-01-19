#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "requests>=2.31",
#   "rich",
#   "openai>=1.0.0",
#   "mlx-lm>=0.30.2"
# ]
# [tool.uv]
# prerelease = "allow"
# ///

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

# Import model provider
from models import get_provider

# --- CONFIGURATION ---
def load_config():
    config_path = Path("config.json")
    default_config = {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "student_model": "liquid/lfm2.5-1.2b",
        "teacher_model": "qwen/qwen3-next-80b",
        "starting_prompt": "You are an event extraction AI. Read the text and output valid JSON.",
        "max_prompt_length": 2000,
        "score_threshold": 9.3,
        "min_iterations": 5,
        "max_iterations": 20
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except Exception as e:
            print(f"Error loading config.json: {e}")
            
    return default_config

CONFIG = load_config()

BASE_URL = CONFIG["base_url"]
API_KEY = CONFIG["api_key"]

# Models
STUDENT_MODEL = CONFIG["student_model"]
TEACHER_MODEL = CONFIG["teacher_model"]

# Constraints
MAX_PROMPT_LENGTH = CONFIG.get("max_prompt_length", 2000)
SCORE_THRESHOLD = CONFIG.get("score_threshold", 9.3)
MIN_ITERATIONS = CONFIG.get("min_iterations", 5)
MAX_ITERATIONS = CONFIG.get("max_iterations", 20)
STARTING_PROMPT = CONFIG.get("starting_prompt", "You are an event extraction AI. Read the text and output valid JSON.")
OPTIMIZATION_TARGET = CONFIG.get("optimization_target", "prompt") # "prompt" or "schema"
STARTING_SCHEMA = CONFIG.get("starting_schema", {
  "type": "object",
  "properties": {
    "events": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "event": {"type": "string", "description": "Short description (5-20 words)"},
          "location": {"type": "string", "description": "Country or region"},
          "severity": {"type": "string", "enum": ["Critical", "High", "Moderate", "Low"]},
          "status": {"type": "string", "enum": ["Ongoing", "Completed", "Emerging"]}
        },
        "required": ["event", "location", "severity", "status"]
      }
    }
  },
  "required": ["events"]
})

# Directories & Files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ARTIFACT_BASE_DIR = Path("optimization_runs")
ARTIFACT_DIR = ARTIFACT_BASE_DIR / f"run_{TIMESTAMP}"
PROMPT_DIR = ARTIFACT_DIR / "prompts"
SCHEMA_DIR = ARTIFACT_DIR / "schemas"
RESPONSE_DIR = ARTIFACT_DIR / "responses"
LOG_FILE = ARTIFACT_DIR / "optimization_log.csv"
BEST_PROMPT_FILE = ARTIFACT_DIR / "best_prompt.txt"
SUMMARY_FILE = ARTIFACT_DIR / "results_summary.txt"

# Initialize Console
console = Console()

# --- 1. STUDENT SCHEMA (Extraction) ---
# Loaded from config now
STUDENT_SCHEMA = STARTING_SCHEMA

# --- 2. TEACHER SCHEMA (Evaluation) ---
TEACHER_SCHEMA = {
  "type": "object",
  "properties": {
    "score": {"type": "integer", "description": "Score from 1-10"},
    "critique": {"type": "string", "description": "Reasoning for the score"},
    "missing_info": {"type": "string", "description": "Specific details missed by the student"}
  },
  "required": ["score", "critique", "missing_info"]
}

# --- 3. META-EVALUATION SCHEMA (Early Stopping) ---
META_EVAL_SCHEMA = {
  "type": "object",
  "properties": {
    "stop_optimization": {"type": "boolean", "description": "Set to true if the prompt is now good enough or if it's no longer improving significantly."},
    "reasoning": {"type": "string", "description": "Reasoning for the decision to stop or continue."}
  },
  "required": ["stop_optimization", "reasoning"]
}

# --- TEST DATA ---
TEST_ARTICLES = [
    # 1. Yemen (Humanitarian)
    """
    By Dale Gavlak * Catholic News Service. AMMAN, Jordan (CNS) -- CAFOD has joined other NGOs in calling for prayer for Yemen.
    After nearly four years of war, more than 14 million people are facing starvation and 85,000 children may have already died. 
    On Jan. 28, Martin Griffiths, U.N. special envoy, pressed for troop withdrawal from Hodeida. 
    "We see immense suffering," said Chris Bain, CEO of CAFOD. Aid workers report rising numbers of displaced civilians.
    """,
    # 2. Brazil (Flood)
    """
    RIO DE JANEIRO (AP) — Heavy rains in southern Brazil have caused a dam to collapse. 
    The death toll has risen to 57. The bursting of the small hydroelectric dam between Cotiporã and Bento Gonçalves sent a two-meter wave of muddy water. 
    Electricity is cut off for nearly 300,000 residents. Governor Eduardo Leite described it as "the worst climate disaster in the state's history."
    """,
    # 3. Germany (Strike)
    """
    BERLIN — Germany’s transport network ground to a halt as the GDL train drivers' union launched a 35-hour nationwide strike. 
    The strike began at 2:00 AM. Deutsche Bahn noted that only 20% of long-distance trains would run. 
    The German Economic Institute estimates the cost at 100 million euros per day.
    """
]

# Initialize Provider
provider = get_provider(CONFIG)

def setup_directories():
    """Creates necessary directories for artifacts and saves used configuration."""
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the configuration used for this run
    config_save_path = ARTIFACT_DIR / "config_used.json"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=2)
    console.print(f"Artifacts will be saved to: [bold]{ARTIFACT_DIR}[/bold]")

def get_completion(model: str, messages: list, target_schema: Optional[dict] = None, temperature: float = 0.1):
    """Handles API calls via the shared provider."""
    return provider.get_completion(model, messages, target_schema, temperature)

def get_batch_completion(model: str, messages_list: List[list], target_schema: Optional[dict] = None, temperature: float = 0.1):
    """Handles batched API calls via the shared provider."""
    return provider.get_batch_completion(model, messages_list, target_schema, temperature)

def save_text(filepath: Path, content: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def ensure_prompt_length(prompt: str, max_length: int) -> str:
    """Iteratively asks the Teacher model to shorten the prompt until it meets the max_length."""
    prompt = prompt.strip().replace('"', '').replace("```", "")
    
    attempts = 0
    max_attempts = 10
    
    while len(prompt) > max_length:
        attempts += 1
        console.print(f"[bold red]Prompt length ({len(prompt)}) exceeds limit ({max_length}). Attempting to shorten (Attempt {attempts}/{max_attempts})...[/bold red]")
        
        shorten_msg = f"""
        The following prompt is too long ({len(prompt)} characters). 
        The maximum allowed length is {max_length} characters.
        
        Please rewrite it to be SIGNIFICANTLY more concise while maintaining all critical instructions.
        
        PROMPT:
        {prompt}
        """
        shortened = get_completion(TEACHER_MODEL, [{"role": "user", "content": shorten_msg}], temperature=0.5)
        
        if shortened:
            prompt = shortened.strip().replace('"', '').replace("```", "")
            if len(prompt) <= max_length:
                console.print(f"[green]Successfully shortened prompt to {len(prompt)} characters.[/green]")
                return prompt
        
        if attempts >= max_attempts:
            console.print(f"[bold red]Failed to shorten prompt after {max_attempts} attempts.[/bold red]")
            return None
            
    return prompt

def generate_summary(history: List[Dict]):
    """
    Asks the Teacher model to summarize the optimization process.
    """
    console.print("\n[bold cyan]Generating Final Summary...[/bold cyan]")
    
    summary_prompt = f"""
    You are an expert Prompt Engineer. Review the history of this optimization session.
    
    HISTORY:
    {json.dumps(history, indent=2)}
    
    Task:
    1. Identify what prompt techniques improved the score.
    2. Identify what techniques caused failures or low scores.
    3. Summarize the best practices for this specific task (Event Extraction).
    
    Output a concise summary.
    """
    
    summary = get_completion(TEACHER_MODEL, [{"role": "user", "content": summary_prompt}])
    if summary:
        save_text(SUMMARY_FILE, summary)
        console.print(Panel(summary, title="Results Summary", border_style="green"))
    else:
        console.print("[red]Failed to generate summary.[/red]")

def run_benchmark():
    setup_directories()
    
    # Initialize Log
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Article_ID", "Student_Output", "Teacher_Score", "Teacher_Critique", "Prompt_Used"])

    # State Tracking
    current_prompt = ensure_prompt_length(STARTING_PROMPT, MAX_PROMPT_LENGTH)
    current_schema = STUDENT_SCHEMA # This is now loaded from CONFIG in lines 80+
    
    if current_prompt is None:
        console.print(f"[bold red][FATAL] Starting prompt exceeds {MAX_PROMPT_LENGTH} characters and could not be shortened. Please check your config.json.[/bold red]")
        return

    global_best_score = -1.0
    global_best_prompt = current_prompt
    global_best_schema = current_schema
    
    # History for Summary
    history = [] 

    console.print(f"[bold green]--- Starting Optimization Loop ---[/bold green]")
    console.print(f"Student: [cyan]{STUDENT_MODEL}[/cyan]")
    console.print(f"Teacher: [cyan]{TEACHER_MODEL}[/cyan]")

    for i in range(MAX_ITERATIONS):
        console.print(f"\n[bold yellow]=== ITERATION {i+1} ===[/bold yellow]")
        
        # 1. Save Current Artifacts
        prompt_path = PROMPT_DIR / f"iter_{i+1}.txt"
        save_text(prompt_path, current_prompt)
        console.print(f"Prompt saved to: {prompt_path} (Length: {len(current_prompt)})")

        schema_path = SCHEMA_DIR / f"iter_{i+1}_schema.json"
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(current_schema, f, indent=2)
        console.print(f"Schema saved to: {schema_path}")

        feedback_bucket = []
        iteration_scores = []

        # 2. STUDENT BATCH GENERATION
        student_msgs_list = [
            [
                {"role": "system", "content": current_prompt},
                {"role": "user", "content": article}
            ]
            for article in TEST_ARTICLES
        ]
        
        console.print(f"  > Batching {len(TEST_ARTICLES)} articles through Student ({STUDENT_MODEL})...")
        # Use current_schema here
        student_outputs = get_batch_completion(STUDENT_MODEL, student_msgs_list, target_schema=current_schema)

        feedback_bucket = []
        iteration_scores = []
        
        # Prepare for Teacher Evaluation
        teacher_msgs_list = []
        valid_indices = []

        for idx, output in enumerate(student_outputs):
            if not output:
                console.print(f"  [red]> Art {idx+1} | Generation failed[/red]")
                continue
            
            # Save Student Answer
            json_path = RESPONSE_DIR / f"iter_{i+1}_art_{idx+1}.json"
            save_text(json_path, output)
            
            article = TEST_ARTICLES[idx]
            eval_instruction = f"""
            Evaluate this extraction based on the text. 
            Article: {article}
            Extraction: {output}
            
            Check for: 1. Factuality 2. Correct Schema 3. Missing Events.
            """
            
            teacher_msgs_list.append([
                {"role": "system", "content": "You are a data auditor."},
                {"role": "user", "content": eval_instruction}
            ])
            valid_indices.append(idx)

        # 3. TEACHER BATCH EVALUATION
        if teacher_msgs_list:
            console.print(f"  > Batching {len(teacher_msgs_list)} outputs through Teacher ({TEACHER_MODEL})...")
            teacher_outputs = get_batch_completion(TEACHER_MODEL, teacher_msgs_list, target_schema=TEACHER_SCHEMA)
            
            for idx, eval_raw in enumerate(teacher_outputs):
                original_idx = valid_indices[idx]
                output = student_outputs[original_idx]
                
                score = 0
                critique = "Error parsing eval"
                
                try:
                    eval_json = json.loads(eval_raw)
                    score = eval_json.get("score", 0)
                    critique = eval_json.get("critique", "")
                    
                    feedback_bucket.append(f"Article {original_idx+1}: {critique}")
                    iteration_scores.append(score)
                    
                    console.print(f"  > Art {original_idx+1} | Score: [bold]{score}[/bold] | {critique[:60]}...")

                    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([i+1, original_idx+1, output, score, critique, current_prompt])
                        
                except Exception as e:
                    console.print(f"    (Eval parsing failed for Art {original_idx+1}: {e})")

        # C. CALCULATE AVERAGES
        avg = sum(iteration_scores)/len(iteration_scores) if iteration_scores else 0
        console.print(f"  > Iteration Average Score: [bold cyan]{avg:.2f}/10[/bold cyan]")

        # D. UPDATE GLOBAL BEST
        if avg > global_best_score:
            global_best_score = avg
            global_best_prompt = current_prompt
            global_best_schema = current_schema
            console.print("  [bold green]> New Best Found![/bold green]")
        
        # Add to history for final summary
        history.append({
            "iteration": i+1,
            "prompt": current_prompt,
            "schema_snippet": str(current_schema)[:100] + "...",
            "avg_score": avg,
            "critiques": feedback_bucket
        })

        # D. META-EVALUATION: Should we stop?
        if i + 1 >= MIN_ITERATIONS:
            console.print("  > Meta-evaluating optimization status...")
            meta_eval_prompt = f"""
            Review the performance of the current prompt in Iteration {i+1}.
            Average Score: {avg:.2f}/10 (Threshold: {SCORE_THRESHOLD})
            Feedback bucket: {json.dumps(feedback_bucket)}
            
            Decide if we should 'stop_optimization'. 
            Stop if:
            1. The average score is >= {SCORE_THRESHOLD}.
            2. The scores have plateaued and major issues are resolved.
            3. The feedback indicates only minor nitpicks remain.
            """
            
            meta_raw = get_completion(TEACHER_MODEL, [{"role": "user", "content": meta_eval_prompt}], target_schema=META_EVAL_SCHEMA)
            try:
                meta_json = json.loads(meta_raw)
                if meta_json.get("stop_optimization", False):
                    console.print(f"  [bold green]> Teacher decided to STOP: {meta_json.get('reasoning')}[/bold green]")
                    break
                else:
                    console.print(f"  [bold blue]> Teacher decided to CONTINUE: {meta_json.get('reasoning')}[/bold blue]")
            except:
                if avg >= SCORE_THRESHOLD:
                    console.print(f"  > Threshold {SCORE_THRESHOLD} reached. Stopping.")
                    break
        else:
            console.print(f"  > (Iteration {i+1} < Min Iterations {MIN_ITERATIONS}. Continuing optimization...)")

        if i < MAX_ITERATIONS - 1: # Don't optimize after the last run
            # E. OPTIMIZE
            console.print(f"  > Optimizing {OPTIMIZATION_TARGET}...")
            
            if OPTIMIZATION_TARGET == "prompt":
                opt_msg = f"""
                The current system prompt is: "{current_prompt}"
                
                It failed on these points in the last round: 
                {json.dumps(feedback_bucket)}
                
                Task: Write a BETTER system prompt to fix these errors.
                Guidelines:
                1. Keep it concise but comprehensive.
                2. Address the specific failures mentioned in the feedback.
                3. Return ONLY the new system prompt text. No "Here is the prompt" or "System Prompt:".
                4. The prompt MUST be under {MAX_PROMPT_LENGTH} characters.
                """
                
                new_prompt = get_completion(TEACHER_MODEL, [{"role": "user", "content": opt_msg}], temperature=0.7)
                if new_prompt:
                    cleaned_prompt = new_prompt.strip().replace('"', '').replace("```", "")
                    
                    # Check length constraint and shorten if needed
                    shortened_prompt = ensure_prompt_length(cleaned_prompt, MAX_PROMPT_LENGTH)
                    
                    if shortened_prompt is None or len(shortened_prompt) > MAX_PROMPT_LENGTH:
                         console.print(f"[bold red]Could not satisfy length constraint ({len(cleaned_prompt) if cleaned_prompt else 'N/A'} chars). Keeping previous valid prompt.[/bold red]")
                         continue

                    current_prompt = shortened_prompt
            
            elif OPTIMIZATION_TARGET == "schema":
                opt_msg = f"""
                The current JSON schema is: 
                {json.dumps(current_schema, indent=2)}
                
                The current system prompt is: "{current_prompt}"
                
                The extraction had these errors in the last round: 
                {json.dumps(feedback_bucket)}
                
                Task: Write a BETTER JSON Schema to fix these errors.
                Guidelines:
                1. You may add fields, change descriptions, or modify enums to enforce better content.
                2. Do not remove core requirements unless they are the cause of the error.
                3. Return ONLY the valid JSON schema. No markdown formatting like ```json or "Here is the schema".
                """
                
                new_schema_str = get_completion(TEACHER_MODEL, [{"role": "user", "content": opt_msg}], temperature=0.7)
                if new_schema_str:
                    try:
                        # Clean up markdown if present
                        if "```json" in new_schema_str:
                            new_schema_str = new_schema_str.split("```json")[1].split("```")[0].strip()
                        elif "```" in new_schema_str:
                            new_schema_str = new_schema_str.split("```")[1].split("```")[0].strip()
                            
                        new_schema = json.loads(new_schema_str)
                        current_schema = new_schema
                        console.print("[green]Successfully optimized schema.[/green]")
                    except json.JSONDecodeError as e:
                        console.print(f"[bold red]Failed to parse new schema: {e}. Keeping previous schema.[/bold red]")
                        continue
                        
    # --- FINALIZATION ---
    console.print("\n[bold]=== OPTIMIZATION COMPLETE ===[/bold]")
    
    # 1. Save Best Artifacts
    if OPTIMIZATION_TARGET == "prompt":
        save_text(BEST_PROMPT_FILE, global_best_prompt)
        console.print(f"Best Prompt (Score: {global_best_score}) saved to: [bold]{BEST_PROMPT_FILE}[/bold]")
    else:
        # Save best schema
        BEST_SCHEMA_FILE = ARTIFACT_DIR / "best_schema.json"
        with open(BEST_SCHEMA_FILE, 'w', encoding='utf-8') as f:
            json.dump(global_best_schema, f, indent=2)
        console.print(f"Best Schema (Score: {global_best_score}) saved to: [bold]{BEST_SCHEMA_FILE}[/bold]")
    
    # 2. Generate Summary
    generate_summary(history)

if __name__ == "__main__":
    run_benchmark()
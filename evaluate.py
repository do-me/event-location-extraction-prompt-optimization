#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "requests>=2.31",
#   "rich",
#   "openai>=1.0.0"
# ]
# ///

import json
import csv
import os
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel

# --- CONFIGURATION ---
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"

# Models
STUDENT_MODEL = "liquid/lfm2.5-1.2b" 
TEACHER_MODEL = "qwen/qwen3-next-80b"

# Directories & Files
ARTIFACT_DIR = Path("optimization_artifacts")
PROMPT_DIR = ARTIFACT_DIR / "prompts"
RESPONSE_DIR = ARTIFACT_DIR / "responses"
LOG_FILE = ARTIFACT_DIR / "optimization_log.csv"
BEST_PROMPT_FILE = "best_prompt.txt"
SUMMARY_FILE = "results_summary.txt"

# Initialize Console
console = Console()

# --- 1. STUDENT SCHEMA (Extraction) ---
STUDENT_SCHEMA = {
  "type": "object",
  "properties": {
    "events": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "event": {"type": "string", "description": "Short description (5-10 words)"},
          "location": {"type": "string", "description": "Country or region"},
          "severity": {"type": "string", "enum": ["Critical", "High", "Moderate", "Low"]},
          "status": {"type": "string", "enum": ["Ongoing", "Completed", "Emerging"]}
        },
        "required": ["event", "location", "severity", "status"]
      }
    }
  },
  "required": ["events"]
}

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

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def setup_directories():
    """Creates necessary directories for artifacts."""
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

def get_completion(model: str, messages: list, target_schema: Optional[dict] = None, temperature: float = 0.1):
    """
    Handles API calls. 
    """
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if target_schema:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response_data",
                "schema": target_schema,
                "strict": False
            }
        }

    try:
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    except Exception as e:
        console.print(f"[bold red][ERROR] Model: {model} | Error: {e}[/bold red]")
        return None

def save_text(filepath: Path, content: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

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
    current_prompt = "You are an event extraction AI. Read the text and output valid JSON."
    
    global_best_score = -1.0
    global_best_prompt = current_prompt
    
    # History for Summary
    history = [] 

    console.print(f"[bold green]--- Starting Optimization Loop ---[/bold green]")
    console.print(f"Student: [cyan]{STUDENT_MODEL}[/cyan]")
    console.print(f"Teacher: [cyan]{TEACHER_MODEL}[/cyan]")

    for i in range(20): # Increased to 20 iterations
        console.print(f"\n[bold yellow]=== ITERATION {i+1} ===[/bold yellow]")
        
        # 1. Save Current Prompt
        prompt_path = PROMPT_DIR / f"iter_{i+1}.txt"
        save_text(prompt_path, current_prompt)
        console.print(f"Prompt saved to: {prompt_path}")

        feedback_bucket = []
        iteration_scores = []

        for idx, article in enumerate(TEST_ARTICLES):
            
            # A. STUDENT GENERATION
            msgs = [
                {"role": "system", "content": current_prompt},
                {"role": "user", "content": article}
            ]
            
            output = get_completion(STUDENT_MODEL, msgs, target_schema=STUDENT_SCHEMA)
            
            if not output: 
                continue 

            # Save Student Answer
            json_path = RESPONSE_DIR / f"iter_{i+1}_art_{idx+1}.json"
            save_text(json_path, output)

            # B. TEACHER EVALUATION
            eval_instruction = f"""
            Evaluate this extraction based on the text. 
            Article: {article}
            Extraction: {output}
            
            Check for: 1. Factuality 2. Correct Schema 3. Missing Events.
            """
            
            eval_msgs = [
                {"role": "system", "content": "You are a data auditor."},
                {"role": "user", "content": eval_instruction}
            ]
            
            eval_raw = get_completion(TEACHER_MODEL, eval_msgs, target_schema=TEACHER_SCHEMA)
            
            score = 0
            critique = "Error parsing eval"
            
            try:
                eval_json = json.loads(eval_raw)
                score = eval_json.get("score", 0)
                critique = eval_json.get("critique", "")
                
                feedback_bucket.append(f"Article {idx+1}: {critique}")
                iteration_scores.append(score)
                
                console.print(f"  > Art {idx+1} | Score: [bold]{score}[/bold] | {critique[:60]}...")

                with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([i+1, idx+1, output, score, critique, current_prompt])
                    
            except Exception as e:
                console.print(f"    (Eval parsing failed: {e})")

        # C. CALCULATE AVERAGES
        avg = sum(iteration_scores)/len(iteration_scores) if iteration_scores else 0
        console.print(f"  > Iteration Average Score: [bold cyan]{avg:.2f}/10[/bold cyan]")

        # D. UPDATE GLOBAL BEST
        if avg > global_best_score:
            global_best_score = avg
            global_best_prompt = current_prompt
            console.print("  [bold green]> New Best Prompt found![/bold green]")
        
        # Add to history for final summary
        history.append({
            "iteration": i+1,
            "prompt": current_prompt,
            "avg_score": avg,
            "critiques": feedback_bucket
        })

        # D. META-EVALUATION: Should we stop?
        console.print("  > Meta-evaluating optimization status...")
        meta_eval_prompt = f"""
        Review the performance of the current prompt in Iteration {i+1}.
        Average Score: {avg:.2f}/10
        Feedback bucket: {json.dumps(feedback_bucket)}
        
        Decide if we should 'stop_optimization'. 
        Stop if:
        1. The average score is >= 9.5.
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
            if avg >= 9.5:
                console.print("  > Threshold reached. Stopping.")
                break

        if i < 19: # Don't optimize after the last run
            # E. OPTIMIZE PROMPT
            console.print("  > Optimizing prompt...")
            opt_msg = f"""
            The current system prompt is: "{current_prompt}"
            
            It failed on these points in the last round: 
            {json.dumps(feedback_bucket)}
            
            Task: Write a BETTER system prompt to fix these errors.
            Guidelines:
            1. Keep it concise but comprehensive.
            2. Address the specific failures mentioned in the feedback.
            3. Return ONLY the new system prompt text. No "Here is the prompt" or "System Prompt:".
            """
            
            new_prompt = get_completion(TEACHER_MODEL, [{"role": "user", "content": opt_msg}], temperature=0.7)
            if new_prompt:
                # Rigorous cleaning
                cleaned_prompt = new_prompt.strip()
                # Remove common wrapper patterns
                for prefix in ["System Prompt:", "New Prompt:", "Updated Prompt:"]:
                    if cleaned_prompt.startswith(prefix):
                        cleaned_prompt = cleaned_prompt[len(prefix):].strip()
                
                current_prompt = cleaned_prompt.replace('"', '').replace("```", "")

    # --- FINALIZATION ---
    console.print("\n[bold]=== OPTIMIZATION COMPLETE ===[/bold]")
    
    # 1. Save Best Prompt
    save_text(BEST_PROMPT_FILE, global_best_prompt)
    console.print(f"Best Prompt (Score: {global_best_score}) saved to: [bold]{BEST_PROMPT_FILE}[/bold]")
    
    # 2. Generate Summary
    generate_summary(history)

if __name__ == "__main__":
    run_benchmark()
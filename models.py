import json
from typing import List, Dict, Optional, Union
from openai import OpenAI
from rich.console import Console

console = Console()

try:
    # IMPORT generate AS WELL
    from mlx_lm import batch_generate, generate, load
    from mlx_lm.sample_utils import make_sampler
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

class ModelProvider:
    def get_completion(self, model: str, messages: List[Dict], target_schema: Optional[dict] = None, temperature: float = 0.1) -> Optional[str]:
        return self.get_batch_completion(model, [messages], target_schema, temperature)[0]

    def get_batch_completion(self, model: str, messages_list: List[List[Dict]], target_schema: Optional[dict] = None, temperature: float = 0.1) -> List[Optional[str]]:
        raise NotImplementedError

class OpenAIProvider(ModelProvider):
    def __init__(self, base_url: str, api_key: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def get_batch_completion(self, model: str, messages_list: List[List[Dict]], target_schema: Optional[dict] = None, temperature: float = 0.1) -> List[Optional[str]]:
        results = []
        for messages in messages_list:
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
                response = self.client.chat.completions.create(**params)
                results.append(response.choices[0].message.content)
            except Exception as e:
                console.print(f"[bold red][ERROR] OpenAI Provider | Model: {model} | Error: {e}[/bold red]")
                results.append(None)
        return results

class MLXProvider(ModelProvider):
    def __init__(self):
        if not HAS_MLX:
            raise ImportError("mlx-lm package is required for MLXProvider but not found. Install with 'pip install mlx-lm'")
        self.cached_models = {} # {checkpoint: (model, tokenizer)}
        self.current_model = None
        self.current_tokenizer = None
        self.current_checkpoint = None

    def _load_model(self, checkpoint: str):
        if self.current_checkpoint != checkpoint:
            if checkpoint in self.cached_models:
                # console.print(f"[bold cyan]MLX Provider: Switching to cached model {checkpoint}...[/bold cyan]")
                self.current_model, self.current_tokenizer = self.cached_models[checkpoint]
                self.current_checkpoint = checkpoint
            else:
                console.print(f"[bold cyan]MLX Provider: Loading model {checkpoint}...[/bold cyan]")
                # Trust remote code is often needed for experimental/new architectures like Qwen3/Mamba
                model, tokenizer = load(path_or_hf_repo=checkpoint, model_config={"trust_remote_code": True})
                self.cached_models[checkpoint] = (model, tokenizer)
                self.current_model = model
                self.current_tokenizer = tokenizer
                self.current_checkpoint = checkpoint

    def get_batch_completion(self, model: str, messages_list: List[List[Dict]], target_schema: Optional[dict] = None, temperature: float = 0.1) -> List[Optional[str]]:
        self._load_model(model)
        
        formatted_prompts = []
        for messages in messages_list:
            if target_schema and messages[-1]["role"] == "user":
                schema_hint = f"\n\nOutput MUST follow this JSON schema: {json.dumps(target_schema)}"
                messages[-1]["content"] += schema_hint

            # apply_chat_template returns a string
            prompt = self.current_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            formatted_prompts.append(prompt)
            
        try:
            sampler = make_sampler(temp=temperature)
            
            # Try Batch Generation first
            results = batch_generate(
                self.current_model, 
                self.current_tokenizer, 
                formatted_prompts, 
                verbose=False,
                max_tokens=2048,
                sampler=sampler
            )
            generated_texts = results.texts
            del results
            return generated_texts

        except AttributeError as e:
            # Catch the MambaCache error specifically
            if "extract" in str(e) or "MambaCache" in str(e):
                console.print(f"[yellow]Batch generation not supported for this model architecture (Mamba/SSM). Falling back to sequential generation.[/yellow]")
                return self._sequential_generate(formatted_prompts, temperature)
            else:
                console.print(f"[bold red][ERROR] MLX Provider | Model: {model} | Batch Error: {e}[/bold red]")
                return [None] * len(messages_list)
        except Exception as e:
            console.print(f"[bold red][ERROR] MLX Provider | Model: {model} | General Error: {e}[/bold red]")
            return [None] * len(messages_list)

    def _sequential_generate(self, prompts: List[str], temperature: float) -> List[Optional[str]]:
        results = []
        # Create sampler once
        sampler = make_sampler(temp=temperature)
        
        for i, prompt in enumerate(prompts):
            try:
                # Use standard generate() for one prompt at a time
                response = generate(
                    self.current_model,
                    self.current_tokenizer,
                    prompt=prompt,
                    verbose=False,
                    max_tokens=2048,
                    sampler=sampler
                )
                results.append(response)
                # print progress for sequential generation as it is slower
                print(f"Processed {i+1}/{len(prompts)} sequentially...")
            except Exception as e:
                console.print(f"[red]Error generating sequence {i}: {e}[/red]")
                results.append(None)
        return results
        
def get_provider(config: dict) -> ModelProvider:
    if config.get("use_mlx", False):
        return MLXProvider()
    else:
        return OpenAIProvider(config["base_url"], config["api_key"])

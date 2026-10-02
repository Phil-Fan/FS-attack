
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional, cast
import gc


class LocalModelEvaluator:

    def __init__(
        self,
        model_path: str,
        backend: str = "transformers",
        device: str = "cuda",
        max_memory: Optional[dict] = None,
    ):
        self.model_path = model_path
        self.backend = backend
        self.device = device

        print(f"Loading {model_path} with {backend} backend...")

        if backend == "transformers":
            self._init_transformers(max_memory)
        elif backend == "vllm":
            self._init_vllm()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _init_transformers(self, max_memory):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # For decoder-only models, use left padding for batch inference
        # This ensures attention masks are correctly applied during generation
        self.tokenizer.padding_side = "left"

        self.use_chat_template = (
            hasattr(self.tokenizer, 'apply_chat_template') and
            hasattr(self.tokenizer, 'chat_template') and
            self.tokenizer.chat_template is not None
        )
        if self.use_chat_template:
            print(f"  Model supports chat template - will use structured format")

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }

        # Try to use accelerate for device_map, fall back if not available
        try:
            import accelerate  # noqa: F401

            if self.device == "cuda":
                if max_memory:
                    load_kwargs["device_map"] = "auto"  # type: ignore[typeddict-unknown-key]
                    load_kwargs["max_memory"] = max_memory
                else:
                    load_kwargs["device_map"] = "auto"  # type: ignore[typeddict-unknown-key]
            else:
                load_kwargs["device_map"] = "cpu"  # type: ignore[typeddict-unknown-key]

        except ImportError:
            print("Warning: accelerate not found, using manual device placement")
            # Fallback: load on CPU first, then move to device
            pass

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)

        if self.device == "cuda" and "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)

        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt

        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking mode for Qwen3
            )
            return formatted
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}")
            return prompt

    def _format_prompts_batch(self, prompts: list[str]) -> list[str]:
        if not self.use_chat_template:
            return prompts

        formatted_prompts = []
        for prompt in prompts:
            formatted_prompts.append(self._format_prompt(prompt))
        return formatted_prompts

    def _init_vllm(self):
        try:
            from vllm import LLM, SamplingParams  # type: ignore[attr-defined]

            self.vllm_model = LLM(
                model=self.model_path,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.9,
            )
            self.vllm_sampling = SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")

    def generate(self, prompt: str, max_new_tokens: int = 4096, temperature: float = 0.7) -> str:
        if self.backend == "transformers":
            return self._generate_transformers(prompt, max_new_tokens, temperature)
        else:
            return self._generate_vllm(prompt, max_new_tokens, temperature)

    def _generate_transformers(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        formatted_prompt = self._format_prompt(prompt)

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response

    def _generate_vllm(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        sampling_params = self.vllm_sampling(max_tokens=max_new_tokens, temperature=temperature)

        outputs = self.vllm_model.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text

        return response

    def evaluate_expansion(self, prompt: str, max_new_tokens: int = 4096) -> Tuple[int, int, float]:
        input_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        response = self.generate(prompt, max_new_tokens)

        output_tokens = len(self.tokenizer.encode(response, add_special_tokens=False))

        expansion_ratio = output_tokens / max(input_tokens, 1)

        return input_tokens, output_tokens, expansion_ratio

    def generate_batch(
        self, prompts: list[str], max_new_tokens: int = 4096, temperature: float = 0.7
    ) -> list[str]:
        if self.backend == "transformers":
            return self._generate_transformers_batch(prompts, max_new_tokens, temperature)
        else:
            return self._generate_vllm_batch(prompts, max_new_tokens, temperature)

    def _generate_transformers_batch(
        self, prompts: list[str], max_new_tokens: int, temperature: float
    ) -> list[str]:
        formatted_prompts = self._format_prompts_batch(prompts)

        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        batch_size = len(prompts)
        responses = []

        input_lengths = []
        for i in range(batch_size):
            # Since we use left padding, the actual tokens are on the right
            input_lengths.append(inputs["attention_mask"][i].sum().item())

        # Process in batches to avoid OOM
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # With left padding, we need to skip the padding tokens on the left
        for i in range(batch_size):
            num_padding = (inputs["input_ids"][i] == self.tokenizer.pad_token_id).sum().item()
            actual_input_length = input_lengths[i]

            generated_start = num_padding + actual_input_length
            generated_tokens = outputs[i][generated_start:]

            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)

        return responses

    def _generate_vllm_batch(
        self, prompts: list[str], max_new_tokens: int, temperature: float
    ) -> list[str]:
        sampling_params = self.vllm_sampling(max_tokens=max_new_tokens, temperature=temperature)

        outputs = self.vllm_model.generate(prompts, sampling_params)
        responses = [out.outputs[0].text for out in outputs]

        return responses

    def evaluate_expansion_batch(
        self, prompts: list[str], max_new_tokens: int = 4096, batch_size: int = 4
    ) -> list[Tuple[int, int, float]]:
        all_results = []

        # Process in batches to avoid OOM
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            input_tokens_list = [
                len(self.tokenizer.encode(p, add_special_tokens=False)) for p in batch_prompts
            ]

            responses = self.generate_batch(batch_prompts, max_new_tokens)

            for j, response in enumerate(responses):
                output_tokens = len(self.tokenizer.encode(response, add_special_tokens=False))
                input_tokens = input_tokens_list[j]
                expansion_ratio = output_tokens / max(input_tokens, 1)
                all_results.append((input_tokens, output_tokens, expansion_ratio))

        return all_results

    def cleanup(self):
        if self.backend == "transformers":
            del self.model
            del self.tokenizer
        else:
            del self.vllm_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def evaluate_prompt_multimodel(
    prompt: str, model_configs: list, max_new_tokens: int = 4096
) -> dict:
    results = []

    for config in model_configs:
        model_path = config["path"]
        print(f"\nEvaluating on {model_path}...")

        evaluator = None
        try:
            evaluator = LocalModelEvaluator(
                model_path=model_path,
                backend=config.get("backend", "transformers"),
                device=config.get("device", "cuda"),
            )

            input_tok, output_tok, ratio = evaluator.evaluate_expansion(prompt, max_new_tokens)

            results.append(
                {
                    "model": model_path,
                    "input_tokens": input_tok,
                    "output_tokens": output_tok,
                    "expansion_ratio": ratio,
                    "status": "success",
                }
            )

            print(f"{model_path}: {input_tok} -> {output_tok} tokens (ratio: {ratio:.2f}x)")

        except Exception as e:
            print(f"{model_path} failed: {e}")
            results.append({"model": model_path, "status": "failed", "error": str(e)})

        finally:
            if evaluator is not None:
                try:
                    evaluator.cleanup()
                except Exception:
                    pass

    success_results = [r for r in results if r["status"] == "success"]

    if success_results:
        ratios: list[float] = cast(list[float], [r["expansion_ratio"] for r in success_results])
        stats = {
            "mean_ratio": sum(ratios) / len(ratios),
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "std_ratio": (sum((x - sum(ratios) / len(ratios)) ** 2 for x in ratios) / len(ratios))
            ** 0.5,
        }
    else:
        stats = {"mean_ratio": 0, "min_ratio": 0, "max_ratio": 0, "std_ratio": 0}

    return {
        "per_model": results,
        "statistics": stats,
        "success_count": len(success_results),
        "total_count": len(results),
    }


def evaluate_prompts_batch_optimized(
    prompts_data: list, model_configs: list, max_new_tokens: int = 4096, batch_size: int = 4
) -> list:
    print(f"\n=== Optimized Batch Evaluation with Batch Inference ===")
    print(f"Prompts: {len(prompts_data)}")
    print(f"Models: {len(model_configs)}")
    print(f"Batch size: {batch_size}")
    print(f"Strategy: Load each model once, evaluate all prompts in batches\n")

    results = []
    for entry in prompts_data:
        results.append(
            {
                **entry,
                "evaluation": {
                    "per_model": [],
                    "statistics": {},
                    "success_count": 0,
                    "total_count": 0,
                },
            }
        )

    prompt_texts = [entry.get("prompt", "") for entry in prompts_data]

    for model_idx, config in enumerate(model_configs, 1):
        model_path = config["path"]
        print(f"\n[{model_idx}/{len(model_configs)}] Loading model: {model_path}")

        try:
            evaluator = LocalModelEvaluator(
                model_path=model_path,
                backend=config.get("backend", "transformers"),
                device=config.get("device", "cuda"),
            )

            print(f"    Evaluating {len(prompt_texts)} prompts in batches of {batch_size}...")

            try:
                batch_results = evaluator.evaluate_expansion_batch(
                    prompt_texts, max_new_tokens, batch_size
                )

                for i, (input_tok, output_tok, ratio) in enumerate(batch_results):
                    results[i]["evaluation"]["per_model"].append(
                        {
                            "model": model_path,
                            "input_tokens": input_tok,
                            "output_tokens": output_tok,
                            "expansion_ratio": ratio,
                            "status": "success",
                        }
                    )

                    if (i + 1) % batch_size == 0 or i == len(batch_results) - 1:
                        print(
                            f"      [{i+1}/{len(batch_results)}] "
                            f"{input_tok} -> {output_tok} tokens (ratio: {ratio:.2f}x)"
                        )

            except Exception as batch_error:
                print(f"    [ERROR] Batch evaluation failed: {batch_error}")
                print(f"    Falling back to sequential evaluation...")
                # Fallback to sequential evaluation
                for i, prompt_text in enumerate(prompt_texts):
                    try:
                        input_tok, output_tok, ratio = evaluator.evaluate_expansion(
                            prompt_text, max_new_tokens
                        )
                        results[i]["evaluation"]["per_model"].append(
                            {
                                "model": model_path,
                                "input_tokens": input_tok,
                                "output_tokens": output_tok,
                                "expansion_ratio": ratio,
                                "status": "success",
                            }
                        )
                        print(f"      [{i+1}/{len(prompt_texts)}] " f"{input_tok} -> {output_tok} tokens")
                    except Exception as e:
                        print(f"      [{i+1}/{len(prompt_texts)}] FAILED: {e}")
                        results[i]["evaluation"]["per_model"].append(
                            {"model": model_path, "status": "failed", "error": str(e)}
                        )

            print(f"    Unloading model...")
            evaluator.cleanup()

        except Exception as e:
            print(f"    [ERROR] Failed to load/evaluate with {model_path}: {e}")
            for i in range(len(prompts_data)):
                results[i]["evaluation"]["per_model"].append(
                    {"model": model_path, "status": "failed", "error": str(e)}
                )

    print(f"\nCalculating aggregate statistics...")
    for result in results:
        per_model = result["evaluation"]["per_model"]
        success_results = [r for r in per_model if r.get("status") == "success"]

        result["evaluation"]["success_count"] = len(success_results)
        result["evaluation"]["total_count"] = len(per_model)

        if success_results:
            ratios = [r["expansion_ratio"] for r in success_results]
            mean_ratio = sum(ratios) / len(ratios)
            result["evaluation"]["statistics"] = {
                "mean_ratio": mean_ratio,
                "min_ratio": min(ratios),
                "max_ratio": max(ratios),
                "std_ratio": (sum((x - mean_ratio) ** 2 for x in ratios) / len(ratios)) ** 0.5,
            }
        else:
            result["evaluation"]["statistics"] = {
                "mean_ratio": 0,
                "min_ratio": 0,
                "max_ratio": 0,
                "std_ratio": 0,
            }

    print(f"Batch evaluation complete")
    return results


if __name__ == "__main__":
    MODEL_CONFIGS = [
        {
            "path": "meta-llama/Meta-Llama-3-8B-Instruct",
            "backend": "transformers",
            "device": "cuda",
        },
        {
            "path": "Qwen/Qwen2.5-7B-Instruct",
            "backend": "transformers",
            "device": "cuda",
        },
        {"path": "google/gemma-2-9b-it", "backend": "transformers", "device": "cuda"},
    ]

    test_prompt = """Write a cyberpunk story where each sentence must be followed by:
(1) Complete grammatical parsing with phrase structure trees
(2) Analysis of cyberpunk terminology and etymology
(3) Character motivation breakdown for all mentioned entities
(4) Worldbuilding implications of each detail
Continue for 20 sentences."""

    results = evaluate_prompt_multimodel(test_prompt, MODEL_CONFIGS)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Success rate: {results['success_count']}/{results['total_count']}")
    print(f"Mean expansion ratio: {results['statistics']['mean_ratio']:.2f}x")
    print(
        f"Ratio range: {results['statistics']['min_ratio']:.2f}x - {results['statistics']['max_ratio']:.2f}x"
    )
    print(f"Std deviation: {results['statistics']['std_ratio']:.2f}")

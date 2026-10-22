import sys
import torch
import os
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Any, Dict, List, Optional, Union, Tuple
from testbed.models.model_base import ModelBase


class QwenVLModelWrapper(ModelBase):
    def __init__(
        self,
        model_root: str,
        processor_class: type, 
        model_class: type,    
        support_models: Optional[List[str]] = None,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            support_models=support_models,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

        # 检查模型结构，用于后续 Hook 注入
        if hasattr(self.model, "visual") and hasattr(self.model, "model"):
            self.internal_llm = self.model.model 
        else:
            print("[QwenVLWrapper] Warning: Standard Qwen-VL structure not found.")

    @staticmethod
    def _looks_like_formatted_prompt(text: str) -> bool:
        markers = (
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
        )
        return any(marker in text for marker in markers)

    def _prepare_prompt(self, text: str, image: Optional[Any]) -> str:
        if self._looks_like_formatted_prompt(text):
            return text

        user_content = []
        if image is not None:
            if isinstance(image, (list, tuple)):
                for single_image in image:
                    user_content.append({"type": "image", "image": single_image})
            else:
                user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": text})

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _think_end_token_id(self) -> Optional[int]:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        token_id = tokenizer.convert_tokens_to_ids("</think>")
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is None or token_id < 0 or (unk_id is not None and token_id == unk_id):
            return None
        return token_id

    def _generate_single_with_budget_forcing(
        self,
        prompt: str,
        image: Optional[Any],
        thinking_budget: int,
        thinking_force_text: str,
        **generate_args,
    ) -> str:
        """Budget-forcing for Qwen3-VL "-Thinking" checkpoints.

        These checkpoints always open the assistant turn with `<think>\\n` and,
        left unconstrained, spend nearly the whole `max_new_tokens` budget
        thinking regardless of task difficulty, whereas plain truncation via
        `max_new_tokens` often cuts a reply off before it ever states a final
        answer. This generates in two passes: stop after `thinking_budget`
        tokens, and if `</think>` hasn't appeared yet, forcibly inject
        `thinking_force_text` (e.g. closing `</think>` + a "final answer"
        nudge) so the remaining budget is spent committing to an answer
        instead of open-ended reasoning.
        """
        processor_inputs = {"text": [prompt], "padding": True, "return_tensors": "pt"}
        if image is not None:
            processor_inputs["images"] = [image]

        inputs = self.processor(**processor_inputs)
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        n_beams = generate_args.pop("num_beams", 1)
        d_sample = generate_args.pop("do_sample", False)
        generate_args.pop("processor_args", None)
        generate_args.pop("model_args", None)
        max_new_tokens = generate_args.pop("max_new_tokens", 4096)

        input_len = inputs["input_ids"].shape[1]
        think_budget = max(1, min(thinking_budget, max_new_tokens))

        def run_generate(gen_inputs: Dict[str, Any], n_new_tokens: int):
            return self.model.generate(
                **gen_inputs,
                num_beams=n_beams,
                do_sample=d_sample,
                max_new_tokens=n_new_tokens,
                **generate_args,
            )

        phase1_ids = run_generate(inputs, think_budget)
        phase1_new_ids = phase1_ids[:, input_len:]
        hit_budget_cap = phase1_new_ids.shape[1] >= think_budget
        remaining_budget = max_new_tokens - think_budget

        if not hit_budget_cap or remaining_budget <= 0:
            all_new_ids = phase1_new_ids
        else:
            end_think_id = self._think_end_token_id()
            already_closed_think = (
                end_think_id is not None and end_think_id in phase1_new_ids[0].tolist()
            )

            forced_ids = None
            if not already_closed_think:
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
                forced_ids = tokenizer(
                    thinking_force_text, return_tensors="pt", add_special_tokens=False,
                )["input_ids"].to(self.device)

            prefix_pieces = [inputs["input_ids"], phase1_new_ids]
            if forced_ids is not None:
                prefix_pieces.append(forced_ids)
            prefix_ids = torch.cat(prefix_pieces, dim=1)

            phase2_remaining = remaining_budget - (
                forced_ids.shape[1] if forced_ids is not None else 0
            )
            if phase2_remaining <= 0:
                all_new_ids = torch.cat(prefix_pieces[1:], dim=1)
            else:
                phase2_inputs = dict(inputs)
                phase2_inputs["input_ids"] = prefix_ids
                phase2_inputs["attention_mask"] = torch.ones_like(prefix_ids)
                phase2_ids = run_generate(phase2_inputs, phase2_remaining)
                phase2_new_ids = phase2_ids[:, prefix_ids.shape[1]:]
                all_new_ids = torch.cat(prefix_pieces[1:] + [phase2_new_ids], dim=1)

        return self.processor.batch_decode(
            all_new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _generate_batch(
        self,
        prompts: List[str],
        images: Optional[List[Any]],
        **generate_args,
    ) -> List[str]:
        processor_inputs = {"text": prompts, "padding": True, "return_tensors": "pt"}
        if images is not None and any(img is not None for img in images):
            processor_inputs["images"] = images

        inputs = self.processor(**processor_inputs)
        final_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        n_beams = generate_args.pop("num_beams", 1)
        d_sample = generate_args.pop("do_sample", False)
        generate_args.pop("processor_args", None)
        generate_args.pop("model_args", None)

        generated_ids = self.model.generate(
            **final_inputs,
            num_beams=n_beams,
            do_sample=d_sample,
            **generate_args,
        )

        input_ids = final_inputs["input_ids"]
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        images: Optional[List[Any]] = None,
        **generate_args,
    ) -> List[str]:
        thinking_budget = generate_args.pop("thinking_budget", None)
        thinking_force_text = generate_args.pop(
            "thinking_force_text",
            "\n</think>\n\n**Final Answer:** \\boxed{",
        )

        if images is None:
            images = [None] * len(prompts)

        formatted_prompts = [
            self._prepare_prompt(text, images[i]) for i, text in enumerate(prompts)
        ]

        try:
            if thinking_budget:
                return [
                    self._generate_single_with_budget_forcing(
                        prompt, image, thinking_budget, thinking_force_text, **dict(generate_args)
                    )
                    for prompt, image in zip(formatted_prompts, images)
                ]

            # Mixed image/no-image batches are easier to handle sample by sample.
            if any(img is None for img in images) and any(img is not None for img in images):
                outputs = []
                for prompt, image in zip(formatted_prompts, images):
                    batch_images = None if image is None else [image]
                    outputs.extend(
                        self._generate_batch([prompt], batch_images, **dict(generate_args))
                    )
                return outputs

            batch_images = None if all(img is None for img in images) else images
            return self._generate_batch(formatted_prompts, batch_images, **generate_args)
        except Exception as e:
            print(f" [GENERATE ERROR]: {e}")
            traceback.print_exc()
            return [""] * len(prompts)

    @property
    def device(self):
        return self.model.device    

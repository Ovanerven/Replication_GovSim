import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .api import AnthropicAPI, AzureOpenAIAPI, MistralAPI, OpenAIAPI, OpenRouter
from .chat import (
    ChatML,
    Cohere,
    DeepSeek,
    Llama3Chat,
    LlamaChat,
    MetaMath,
    MistralInstruct,
    MixtralInstruct,
    Model,
    Phi3,
    Phi4,
    Vicuna,
    Gemma,
    Qwen,
    DeepSeek_Llama,
    DeepSeek_Qwen,
)
from .model import Model


def get_api_model(name, seed):
    if "z-gpt" in name.lower():
        return AzureOpenAIAPI(name, seed)
    if "gpt" in name.lower():
        return OpenAIAPI(name, seed)
    elif "mistral" in name.lower():
        return MistralAPI(name, seed)
    elif "claude" in name.lower():
        return AnthropicAPI(name, seed)
    elif "openrouter" in name.lower():
        name = name.replace("openrouter-", "")
        return OpenRouter(name, seed)
    else:
        raise ValueError(f"Unknown model name {name}")


def get_model(name, is_api=False, seed=42, backend_name="transformers"):
    if is_api:
        return get_api_model(name, seed)
    trust_remote_code = False
    use_fast = True
    extend_context_length = True
    #Phi4 fix:
    use_flash_attention = False

    if "metamath" in name.lower():
        cls = MetaMath
    elif "gemma" in name.lower():
        cls = Gemma
    elif "deepseek" in name.lower() and 'llama' in name.lower():
        cls = DeepSeek_Llama
        trust_remote_code = True  # Add this
    elif "llama-3" in name.lower():
        cls = Llama3Chat
    elif "llama" in name.lower():
        cls = LlamaChat
    elif "phi-3" in name.lower():
        cls = Phi3
        trust_remote_code = True
    elif "phi-4" in name.lower():
        cls = Phi4
        trust_remote_code = True
        use_flash_attention = True
    elif "smaug" in name.lower():
        cls = LlamaChat
    elif "vicuna" in name.lower():
        cls = Vicuna
    elif "wizardlm" in name.lower():
        cls = Vicuna
    elif "pro-mistral" in name.lower():
        cls = ChatML
    elif "dbrx-instruct" in name.lower():
        trust_remote_code = True
        cls = ChatML
    elif "mistral" in name.lower() and "instruct" in name.lower():
        cls = MistralInstruct
    elif "mixtral" in name.lower() and "instruct" in name.lower():
        cls = MixtralInstruct
    elif "hermes-2-mixtral" in name.lower() or "qwen" in name.lower():
        cls = ChatML
    elif "deepseek-qwen" in name.lower():
        cls = DeepSeek_Qwen
        trust_remote_code = True
    elif "deepseek" in name.lower():
        cls = DeepSeek
    # elif "command" in name.lower():
        cls = Cohere
    elif "qwen2.5-math" in name.lower():
        cls = Qwen
    else:
        raise ValueError(f"Unknown model name {name}")
    
    print("Selected template class:", cls.__name__)

    branch = "main"
    if "gptq" in name.lower():
        branch = "gptq-4bit-32g-actorder_True"

        if "metamath" in name.lower() and not "mistral" in name.lower():
            branch = "gptq-4-32g-actorder_True"
        elif "smaug" in name.lower():
            branch = "main"
        elif name.startswith("Qwen"):
            branch = "main"
            extend_context_length = False

        if not "thebloke" in name.lower():
            branch = "main"
            extend_context_length = False

    tokenizer = AutoTokenizer.from_pretrained(
        name, use_fast=use_fast, trust_remote_code=trust_remote_code
    )

    if backend_name == "transformers":
        model_config = AutoConfig.from_pretrained(
            name, 
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2" if use_flash_attention else None
        )

        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="balanced",
            trust_remote_code=trust_remote_code,
            revision=branch,
            # Phi4 fix:
            attn_implementation="flash_attention_2" if use_flash_attention else None,
            # original:
            # attn_implementation=(
            #     "flash_attention_2" if not "gptq" in name.lower() else None
            # ),
            torch_dtype=(
                model_config.torch_dtype if not "gptq" in name.lower() else None
            ),
        )

        if "gptq" in name.lower() and extend_context_length:
            from auto_gptq import exllama_set_max_input_length

            model = exllama_set_max_input_length(model, max_input_length=4096)

        model = torch.compile(model)
        backend = Model(
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            template=cls().template,
        )
    elif backend_name == "vllm":
        from .vllm import ModelVLLMBackend

        backend = ModelVLLMBackend(
            name, tokenizer, trust_remote_code, cls().template, seed
        )

    return backend
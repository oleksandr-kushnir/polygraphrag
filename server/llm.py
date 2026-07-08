"""OpenAI-compatible LLM / vision / embedding shims handed to LightRAG and RAG-Anything.

Each reads its endpoint configuration live from server.config (rather than import-time copies),
so a monkeypatch or runtime reconfig of the config module takes effect on the next call. The
text LLM additionally routes by phase (extraction vs query) via config._active_llm_cfg.
"""

import logging

from server import config


async def _llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    import openai

    model, base_url, api_key, is_openai = config._active_llm_cfg()
    logging.debug("llm call: phase=%s model=%s", config._llm_phase.get(), model)
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        **config._llm_call_kwargs(kwargs, is_openai),
    )
    return resp.choices[0].message.content


async def _vision_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    image_data=None,
    messages=None,
    **kwargs,
):
    import openai

    client = openai.AsyncOpenAI(api_key=config.VISION_API_KEY, base_url=config.VISION_BASE_URL)
    if messages is not None:
        final_messages = messages
    elif image_data is not None:
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "text", "text": prompt},
        ]
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(history_messages)
        final_messages.append({"role": "user", "content": content})
    else:
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(history_messages)
        final_messages.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(
        model=config.VISION_MODEL,
        messages=final_messages,
        **config._llm_call_kwargs(kwargs, is_openai=config._VISION_IS_OPENAI),
    )
    return resp.choices[0].message.content


async def _embedding_func(texts: list[str]):
    import numpy as np
    import openai

    client = openai.AsyncOpenAI(api_key=config.EMBEDDING_API_KEY, base_url=config.EMBEDDING_BASE_URL)
    resp = await client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data])

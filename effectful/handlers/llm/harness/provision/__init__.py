"""Handlers that bind the agent loop to a model backend.

Every model round ends in one operation, :func:`~effectful.handlers.llm.harness.hooks.completion`, whose
default rule calls ``litellm.completion`` directly. A provision handler
intercepts it to fix which model is called and how. The package has one:
:class:`~effectful.handlers.llm.harness.provision.litellm.LiteLLMConfigurer`, which :func:`~effectful.handlers.llm.harness.harness`
installs with ``model=`` and every keyword it does not itself recognise
(``tool_choice``, ``reasoning_effort``, ``num_retries``, ...).

Scope a different model to part of a program by installing another configurer
inside the harness; the innermost ``model`` wins and other settings merge::

    with handler(LiteLLMConfigurer(model="openai/gpt-4.1-mini")):
        draft = writer.write(request)

What the configurer adds to each request and enforces on each response is in
:mod:`~effectful.handlers.llm.harness.provision.litellm`.
"""

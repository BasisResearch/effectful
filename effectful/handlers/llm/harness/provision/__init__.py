"""Handlers that bind the agent loop to a model backend.

Every model round ends in one operation, :func:`~effectful.handlers.llm.harness.hooks.completion`, whose
default rule calls ``litellm.completion`` directly. A provision handler
intercepts it to fix which model is called and how. The package has one:
:class:`~effectful.handlers.llm.harness.provision.litellm.LiteLLMConfigurer`.
:func:`~effectful.handlers.llm.harness.harness` passes ``model=`` and provider
options such as ``tool_choice`` and ``reasoning_effort`` to it. The recognized
``num_retries`` harness option configures both LiteLLM transport retries and
the harness's retry handler for rejected answers.

Scope a different model to part of a program by installing another configurer
inside the harness; the innermost ``model`` wins and other settings merge. For
example, inside a harness where ``writer`` and ``request`` already exist::

    from effectful.handlers.llm.harness.provision.litellm import LiteLLMConfigurer
    from effectful.ops.semantics import handler

    with handler(LiteLLMConfigurer(model="openai/gpt-4.1-mini")):
        draft = writer.write(request)

What the configurer adds to each request and enforces on each response is in
:mod:`~effectful.handlers.llm.harness.provision.litellm`.
"""

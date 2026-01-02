Contextual LLM Agents
======================
Here we give an example of using effectful to implement chatbot-style context-aware LLM agents.

In the code below, we define a helper class :class:`Agent` which wraps its
subclasses' template operations in a wrapper that stores and persists
the history of prior interactions with the LLM:
  - :func:`_format_model_input` wraps every prompt sent to the LLM and
    stashes the generated API message into a state variable.
  - :func:`_compute_response` wraps the response from the LLM provider and
    stashes the returned message into the state.

Using this we can construct an agent which remembers the context of
the conversation:

.. literalinclude:: ./agent.py
    :language: python


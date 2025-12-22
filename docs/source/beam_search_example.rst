Angelic Nondeterminism
======================

Here we give an example of *angelic nondeterminism* in effectful [#f1]_.
Our model is a nondeterministic program that makes choices using a ``choose`` effect and uses a ``score`` effect to sum up a final score.
We implement a beam search that optimizes this final score as a handler for the ``choose`` and ``score`` effects.

The beam search works by running the model until it reaches a ``choose``, at which point the continuation is captured.
This continuation is resumed multiple times with different values from ``choose`` to expand the beam.
The intermediate score is used to rank the beam candidates.

Because Python does not have support for first-class continuations, we use *thermometer continuations* [#f2]_.
A thermometer continuation works by tracking any nondeterminism
(essentially, the model is rerun from the start replaying the ``choose`` effects).
If ``choose`` is the only source of nondeterminism, then the 
after each ``choose`` and replaying it  uses *thermometer continuations* to 

.. literalinclude:: ./beam.py
    :language: python

References
----------

.. [#f1] Li, Z., Solar-Lezama, A., Yue, Y., and Zheng, S., "EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths", 2025. https://arxiv.org/abs/2512.03571

.. [#f2] James Koppel, Gabriel Scherer, and Armando Solar-Lezama. 2018. Capturing the future by replaying the past (functional pearl). Proc. ACM Program. Lang. 2, ICFP, Article 76 (September 2018), 29 pages. https://doi.org/10.1145/3236771

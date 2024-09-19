# import pytest

# UNIMPLEMENTED_SUBSTRINGS = [
#     "infer.JitTrace_ELBO",
#     "the event_dim arg",
#     "optim.ClippedAdam",
#     "infer.TraceMeanField_ELBO",
# ]


# def pytest_runtest_call(item):
#     try:
#         item.runtest()
#     except NotImplementedError as e:
#         if any(s in str(e) for s in UNIMPLEMENTED_SUBSTRINGS):
#             pytest.xfail(str(e))
#         else:
#             raise e

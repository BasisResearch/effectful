"""DS-1000 scipy problems and their execution oracle, for `ds1000.py`.

Data and no skills, deliberately: the harness splices a skill's defining
module's source into its system prompt, and each problem's ``code_context``
contains the DS-1000 *reference solution* (inside ``generate_ans``) alongside
the test harness. Keeping problems in a module the solver never shares -- it
is imported by ``ds1000.py``, so the solver's prompt lists only this module's
*name* -- is what makes the held-out evaluation honest (the module-scoping
lesson of ``autoformalization/auditing.py``).

Problems are the scipy subset curated by `strands-labs/ai-functions`
(Apache-2.0) for its ``memory_backprop_scipy`` demo, taken verbatim from the
DS-1000 benchmark ("DS-1000: A Natural and Reliable Benchmark for Data Science
Code Generation", Lai et al., arXiv:2211.11501; problems derive from
StackOverflow, CC-BY-SA-4.0). Each has an ``id``, ``library``, a ``prompt``
(the question with a ``BEGIN SOLUTION`` marker), and a ``code_context`` (the
executable test harness that inserts the candidate and asserts correctness).
``scipy_716`` in the training set teaches the ``minimize()`` contract the
held-out test problem needs.

The oracle below is pure Python -- no model sits in the scoring path. It
``exec``s model-written code by design (the same caveat
``choreographies/multi_agent_choreography.py`` carries): fine against these
fixed benchmark problems, but worth knowing before pointing it elsewhere.
"""

from typing import Any

TRAIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "scipy_711",
        "library": "Scipy",
        "prompt": "Problem:\nI have a set of data and I want to compare which line describes it best (polynomials of different orders, exponential or logarithmic).\nI use Python and Numpy and for polynomial fitting there is a function polyfit(). \nHow do I fit y = Alogx + B using polyfit()? The result should be an np.array of [A, B]\nA:\n<code>\nimport numpy as np\nimport scipy\nx = np.array([1, 7, 20, 50, 79])\ny = np.array([10, 19, 30, 35, 51])\n\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\n\n\ndef generate_test_case(test_case_id):\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            x = np.array([1, 7, 20, 50, 79])\n            y = np.array([10, 19, 30, 35, 51])\n        return x, y\n\n    def generate_ans(data):\n        _a = data\n        x, y = _a\n        result = np.polyfit(np.log(x), y, 1)\n        return result\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    assert np.allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport numpy as np\nimport scipy\nx, y = test_input\n[insert]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_712",
        "library": "Scipy",
        "prompt": "Problem:\nI have a set of data and I want to compare which line describes it best (polynomials of different orders, exponential or logarithmic).\nI use Python and Numpy and for polynomial fitting there is a function polyfit(). \nHow do I fit y = A + Blogx using polyfit()? The result should be an np.array of [A, B]\nA:\n<code>\nimport numpy as np\nimport scipy\nx = np.array([1, 7, 20, 50, 79])\ny = np.array([10, 19, 30, 35, 51])\n\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\n\n\ndef generate_test_case(test_case_id):\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            x = np.array([1, 7, 20, 50, 79])\n            y = np.array([10, 19, 30, 35, 51])\n        return x, y\n\n    def generate_ans(data):\n        _a = data\n        x, y = _a\n        result = np.polyfit(np.log(x), y, 1)[::-1]\n        return result\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    assert np.allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport numpy as np\nimport scipy\nx, y = test_input\n[insert]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_713",
        "library": "Scipy",
        "prompt": "Problem:\nI have a set of data and I want to compare which line describes it best (polynomials of different orders, exponential or logarithmic).\nI use Python and Numpy and for polynomial fitting there is a function polyfit(). But I found no such functions for exponential and logarithmic fitting.\nHow do I fit y = A*exp(Bx) + C ? The result should be an np.array of [A, B, C]. I know that polyfit performs bad for this function, so I would like to use curve_fit to solve the problem, and it should start from initial guess p0.\nA:\n<code>\nimport numpy as np\nimport scipy.optimize\ny = np.array([1, 7, 20, 50, 79])\nx = np.array([10, 19, 30, 35, 51])\np0 = (4, 0.1, 1)\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\nimport scipy.optimize\n\n\ndef generate_test_case(test_case_id):\n\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            y = np.array([1, 7, 20, 50, 79])\n            x = np.array([10, 19, 30, 35, 51])\n            p0 = (4, 0.1, 1)\n        return x, y, p0\n\n    def generate_ans(data):\n        _a = data\n        x, y, p0 = _a\n        result = scipy.optimize.curve_fit(\n            lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=p0\n        )[0]\n        return result\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    assert np.allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport numpy as np\nimport scipy.optimize\nx, y, p0 = test_input\n[insert]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_714",
        "library": "Scipy",
        "prompt": "Problem:\nI can't figure out how to do a Two-sample KS test in Scipy.\nAfter reading the documentation scipy kstest\nI can see how to test where a distribution is identical to standard normal distribution\nfrom scipy.stats import kstest\nimport numpy as np\nx = np.random.normal(0,1,1000)\ntest_stat = kstest(x, 'norm')\n#>>> test_stat\n#(0.021080234718821145, 0.76584491300591395)\nWhich means that at p-value of 0.76 we can not reject the null hypothesis that the two distributions are identical.\nHowever, I want to compare two distributions and see if I can reject the null hypothesis that they are identical, something like:\nfrom scipy.stats import kstest\nimport numpy as np\nx = np.random.normal(0,1,1000)\nz = np.random.normal(1.1,0.9, 1000)\nand test whether x and z are identical\nI tried the naive:\ntest_stat = kstest(x, z)\nand got the following error:\nTypeError: 'numpy.ndarray' object is not callable\nIs there a way to do a two-sample KS test in Python? If so, how should I do it?\nThank You in Advance\nA:\n<code>\nfrom scipy import stats\nimport numpy as np\nnp.random.seed(42)\nx = np.random.normal(0, 1, 1000)\ny = np.random.normal(0, 1, 1000)\n</code>\nstatistic, p_value = ... # put solution in these variables\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\nfrom scipy import stats\n\n\ndef generate_test_case(test_case_id):\n\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            np.random.seed(42)\n            x = np.random.normal(0, 1, 1000)\n            y = np.random.normal(0, 1, 1000)\n        elif test_case_id == 2:\n            np.random.seed(42)\n            x = np.random.normal(0, 1, 1000)\n            y = np.random.normal(1.1, 0.9, 1000)\n        return x, y\n\n    def generate_ans(data):\n        _a = data\n        x, y = _a\n        statistic, p_value = stats.ks_2samp(x, y)\n        return [statistic, p_value]\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    np.testing.assert_allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nfrom scipy import stats\nimport numpy as np\nnp.random.seed(42)\nx, y = test_input\n[insert]\nresult = [statistic, p_value]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(2):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_715",
        "library": "Scipy",
        "prompt": "Problem:\nI can't figure out how to do a Two-sample KS test in Scipy.\nAfter reading the documentation scipy kstest\nI can see how to test where a distribution is identical to standard normal distribution\nfrom scipy.stats import kstest\nimport numpy as np\nx = np.random.normal(0,1,1000)\ntest_stat = kstest(x, 'norm')\n#>>> test_stat\n#(0.021080234718821145, 0.76584491300591395)\nWhich means that at p-value of 0.76 we can not reject the null hypothesis that the two distributions are identical.\nHowever, I want to compare two distributions and see if I can reject the null hypothesis that they are identical, something like:\nfrom scipy.stats import kstest\nimport numpy as np\nx = np.random.normal(0,1,1000)\nz = np.random.normal(1.1,0.9, 1000)\nand test whether x and z are identical\nI tried the naive:\ntest_stat = kstest(x, z)\nand got the following error:\nTypeError: 'numpy.ndarray' object is not callable\nIs there a way to do a two-sample KS test in Python, then test whether I can reject the null hypothesis that the two distributions are identical(result=True means able to reject, and the vice versa) based on alpha? If so, how should I do it?\nThank You in Advance\nA:\n<code>\nfrom scipy import stats\nimport numpy as np\nnp.random.seed(42)\nx = np.random.normal(0, 1, 1000)\ny = np.random.normal(0, 1, 1000)\nalpha = 0.01\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\nfrom scipy import stats\n\n\ndef generate_test_case(test_case_id):\n\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            np.random.seed(42)\n            x = np.random.normal(0, 1, 1000)\n            y = np.random.normal(0, 1, 1000)\n        elif test_case_id == 2:\n            np.random.seed(42)\n            x = np.random.normal(0, 1, 1000)\n            y = np.random.normal(1.1, 0.9, 1000)\n        alpha = 0.01\n        return x, y, alpha\n\n    def generate_ans(data):\n        _a = data\n        x, y, alpha = _a\n        s, p = stats.ks_2samp(x, y)\n        result = p <= alpha\n        return result\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    np.testing.assert_array_equal(result, ans)\n    return 1\n\n\nexec_context = r"""\nfrom scipy import stats\nimport numpy as np\nx, y, alpha = test_input\n[insert]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(2):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_716",  # KEY training example — teaches correct minimize() contract
        "library": "Scipy",
        "prompt": "Problem:\nAccording to the SciPy documentation it is possible to minimize functions with multiple variables, yet it doesn't tell how to optimize on such functions.\nfrom scipy.optimize import minimize\nfrom math import sqrt, sin, pi, cos\ndef f(c):\n  return sqrt((sin(pi/2) + sin(0) + sin(c) - 2)**2 + (cos(pi/2) + cos(0) + cos(c) - 1)**2)\nprint minimize(f, 3.14/2 + 3.14/7)\n\nThe above code does try to minimize the function f, but for my task I need to minimize with respect to three variables, starting from `initial_guess`.\nSimply introducing a second argument and adjusting minimize accordingly yields an error (TypeError: f() takes exactly 2 arguments (1 given)).\nHow does minimize work when minimizing with multiple variables.\nI need to minimize f(a,b,c)=((a+b-c)-2)**2 + ((3*a-b-c))**2 + sin(b) + cos(b) + 4.\nResult should be a list=[a,b,c], the parameters of minimized function.\n\nA:\n<code>\nimport scipy.optimize as optimize\nfrom math import sqrt, sin, pi, cos\n\ninitial_guess = [-1, 0, -3]\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\nfrom scipy import optimize\n\n\ndef generate_test_case(test_case_id):\n\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            a = [-1, 0, -3]\n        return a\n\n    def generate_ans(data):\n        _a = data\n        initial_guess = _a\n\n        def g(params):\n            a, b, c = params\n            return (\n                ((a + b - c) - 2) ** 2\n                + ((3 * a - b - c)) ** 2\n                + np.sin(b)\n                + np.cos(b)\n                + 4\n            )\n\n        res = optimize.minimize(g, initial_guess)\n        result = res.x\n        return result\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    def g(params):\n        a, b, c = params\n        return (\n            ((a + b - c) - 2) ** 2 + ((3 * a - b - c)) ** 2 + np.sin(b) + np.cos(b) + 4\n        )\n\n    assert abs(g(result) - g(ans)) < 1e-2\n    return 1\n\n\nexec_context = r"""\nimport scipy.optimize as optimize\nfrom math import sqrt, sin, pi, cos\ninitial_guess = test_input\n[insert]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_717",
        "library": "Scipy",
        "prompt": "Problem:\nHow does one convert a list of Z-scores from the Z-distribution (standard normal distribution, Gaussian distribution) to left-tailed p-values? I have yet to find the magical function in Scipy's stats module to do this, but one must be there.\nA:\n<code>\nimport numpy as np\nimport scipy.stats\nz_scores = np.array([-3, -2, 0, 2, 2.5])\n</code>\np_values = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\nimport scipy.stats\n\n\ndef generate_test_case(test_case_id):\n\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            a = np.array([-3, -2, 0, 2, 2.5])\n        return a\n\n    def generate_ans(data):\n        _a = data\n        z_scores = _a\n        temp = np.array(z_scores)\n        p_values = scipy.stats.norm.cdf(temp)\n        return p_values\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    np.testing.assert_allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport numpy as np\nimport scipy.stats\nz_scores = test_input\n[insert]\nresult = p_values\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_718",
        "library": "Scipy",
        "prompt": "Problem:\nHow does one convert a list of Z-scores from the Z-distribution (standard normal distribution, Gaussian distribution) to left-tailed p-values? Original data is sampled from X ~ N(mu, sigma). I have yet to find the magical function in Scipy's stats module to do this, but one must be there.\nA:\n<code>\nimport scipy.stats\nimport numpy as np\nz_scores = [-3, -2, 0, 2, 2.5]\nmu = 3\nsigma = 4\n</code>\np_values = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport scipy\nfrom scipy import sparse\nimport scipy.stats\nimport copy\nimport io\nfrom scipy import integrate\n\n\ndef generate_test_case(test_case_id):\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            z_scores = [-3, -2, 0, 2, 2.5]\n            mu = 3\n            sigma = 4\n        return z_scores, mu, sigma\n\n    def generate_ans(data):\n        _a = data\n        z_scores, mu, sigma = _a\n        temp = np.array(z_scores)\n        p_values = scipy.stats.norm.cdf(temp)\n        return p_values\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    np.testing.assert_allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport numpy as np\nimport scipy.stats\nz_scores, mu, sigma = test_input\n[insert]\nresult = p_values\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
]

# 1 test problem: scipy_787
TEST_PROBLEMS = [
    {
        "id": "scipy_787",
        "library": "Scipy",
        "prompt": "Problem:\n\n\nI am having a problem with minimization procedure. Actually, I could not create a correct objective function for my problem.\nProblem definition\n•\tMy function: yn = a_11*x1**2 + a_12*x2**2 + ... + a_m*xn**2,where xn- unknowns, a_m - coefficients. n = 1..N, m = 1..M\n•\tIn my case, N=5 for x1,..,x5 and M=3 for y1, y2, y3.\nI need to find the optimum: x1, x2,...,x5 so that it can satisfy the y\nMy question:\n•\tHow to solve the question using scipy.optimize?\nMy code:   (tried in lmfit, but return errors. Therefore I would ask for scipy solution)\nimport numpy as np\nfrom lmfit import Parameters, minimize\ndef func(x,a):\n    return np.dot(a, x**2)\ndef residual(pars, a, y):\n    vals = pars.valuesdict()\n    x = vals['x']\n    model = func(x,a)\n    return (y - model)**2\ndef main():\n    # simple one: a(M,N) = a(3,5)\n    a = np.array([ [ 0, 0, 1, 1, 1 ],\n                   [ 1, 0, 1, 0, 1 ],\n                   [ 0, 1, 0, 1, 0 ] ])\n    # true values of x\n    x_true = np.array([10, 13, 5, 8, 40])\n    # data without noise\n    y = func(x_true,a)\n    #************************************\n    # Apriori x0\n    x0 = np.array([2, 3, 1, 4, 20])\n    fit_params = Parameters()\n    fit_params.add('x', value=x0)\n    out = minimize(residual, fit_params, args=(a, y))\n    print out\nif __name__ == '__main__':\nmain()\nResult should be optimal x array. The method I hope to use is L-BFGS-B, with added lower bounds on x.\n\nA:\n\n\n<code>\nimport scipy.optimize\nimport numpy as np\nnp.random.seed(42)\na = np.random.rand(3,5)\nx_true = np.array([10, 13, 5, 8, 40])\ny = a.dot(x_true ** 2)\nx0 = np.array([2, 3, 1, 4, 20])\nx_lower_bounds = x_true / 2\n</code>\nout = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\nimport scipy.optimize\n\n\ndef generate_test_case(test_case_id):\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            np.random.seed(42)\n            a = np.random.rand(3, 5)\n            x_true = np.array([10, 13, 5, 8, 40])\n            y = a.dot(x_true**2)\n            x0 = np.array([2, 3, 1, 4, 20])\n            x_bounds = x_true / 2\n        return a, x_true, y, x0, x_bounds\n\n    def generate_ans(data):\n        _a = data\n        a, x_true, y, x0, x_lower_bounds = _a\n\n        def residual_ans(x, a, y):\n            s = ((y - a.dot(x**2)) ** 2).sum()\n            return s\n\n        bounds = [[x, None] for x in x_lower_bounds]\n        out = scipy.optimize.minimize(\n            residual_ans, x0=x0, args=(a, y), method="L-BFGS-B", bounds=bounds\n        ).x\n        return out\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    assert np.allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport scipy.optimize\nimport numpy as np\na, x_true, y, x0, x_lower_bounds = test_input\n[insert]\nresult = out\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
]


# ── Execution oracle (pure Python, no model calls) ───────────────────────────


import dataclasses
import signal
import threading
import traceback
from contextlib import contextmanager


@dataclasses.dataclass
class ExecutionResult:
    """Outcome of running one candidate solution against a problem's tests."""

    passed: bool
    error: str | None = None
    test_input: str | None = None
    expected_output: str | None = None
    actual_output: str | None = None


@contextmanager
def _timeout(seconds: int = 30):
    """Raise ``TimeoutError`` if the wrapped block runs longer than ``seconds``.

    SIGALRM only works on the main thread; elsewhere the block runs unguarded.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _truncated(obj: object, max_len: int = 200) -> str:
    s = repr(obj)
    return s[:max_len] + "..." if len(s) > max_len else s


def extract_solution_code(raw_output: str) -> str:
    """Strip markdown fences and DS-1000 solution markers from model output."""
    text = raw_output
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end]
    elif "```" in text:
        start = text.index("```") + 3
        newline = text.find("\n", start)
        if newline != -1:
            start = newline + 1
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end]
    for marker in ("BEGIN SOLUTION", "END SOLUTION", "<code>", "</code>"):
        text = text.replace(marker, "")
    return text


def _assertion_detail(
    solution_code: str, code_context: str
) -> tuple[str | None, str | None, str | None]:
    """Re-run a failed solution to capture (test input, expected, actual)."""
    try:
        with _timeout(10):
            test_env: dict = {}
            exec(code_context, test_env)  # noqa: S102
            exec_context = test_env.get("exec_context", "")
            generate_test_case = test_env.get("generate_test_case")
            if not exec_context or not generate_test_case:
                return None, None, None
            test_input, expected = generate_test_case(1)
            run_env: dict = {"test_input": test_input}
            exec(exec_context.replace("[insert]", solution_code), run_env)  # noqa: S102
            return (
                _truncated(test_input),
                _truncated(expected),
                _truncated(run_env.get("result")),
            )
    except Exception:  # noqa: BLE001 -- best-effort detail capture
        return None, None, None


def execute_and_test(solution_code: str, code_context: str) -> ExecutionResult:
    """Run the DS-1000 test harness against a candidate solution.

    The benchmark's own preamble is executed first and *outside* the verdict's
    ``except``: a failure there is a broken problem definition, not a wrong
    answer, and scoring the two the same way makes an unimportable module in
    `code_context` read as the model failing every attempt at that problem --
    with the resulting FAIL then backpropagated as if it were the model's to
    learn from.
    """
    test_env: dict = {}
    try:
        exec(code_context, test_env)  # noqa: S102
    except Exception as e:
        raise RuntimeError(f"benchmark preamble failed to execute: {e}") from e

    try:
        with _timeout(30):
            test_env["test_execution"](solution_code)
            return ExecutionResult(passed=True)
    except TimeoutError as e:
        return ExecutionResult(passed=False, error=str(e))
    except AssertionError as e:
        test_input, expected, actual = _assertion_detail(solution_code, code_context)
        return ExecutionResult(
            passed=False,
            error=f"Test assertion failed: {e}" if str(e) else "Test assertion failed",
            test_input=test_input,
            expected_output=expected,
            actual_output=actual,
        )
    except Exception as e:  # noqa: BLE001 -- any solution error is a failure, captured as feedback
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        short = "".join(tb[-3:]) if len(tb) > 3 else "".join(tb)
        return ExecutionResult(passed=False, error=f"{type(e).__name__}: {e}\n{short}")


def build_feedback(problem: dict, solution: str, result: ExecutionResult) -> str:
    """The oracle's verdict as optimizer feedback, one string per problem."""
    if result.passed:
        return (
            f"[{problem['library']}] {problem['id']} SOLVED.\n"
            f"Working solution:\n{solution}\n"
            f"Remember this pattern for similar future problems."
        )
    parts = [f"Error: {result.error}"]
    if result.test_input:
        parts.append(f"Test input: {result.test_input}")
    if result.expected_output:
        parts.append(f"Expected output: {result.expected_output}")
    if result.actual_output:
        parts.append(f"Actual output: {result.actual_output}")
    return (
        f"[{problem['library']}] {problem['id']} FAILED.\n"
        f"Your code:\n{solution}\n" + "\n".join(parts)
    )

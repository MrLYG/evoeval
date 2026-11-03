# Changes to upstream files

The base is upstream `evo-eval/evoeval` at `d5ca310`. Its evaluation code is identical to `evoeval==0.1.0`. Besides adding `harbor_parity/`, this branch changes exactly one upstream file, as a scoring bugfix. The Harbor adapter applies the same fix.

## `evoeval/eval_test/__init__.py`: float values must have the same number of elements

`is_equal` compares float values (floats, or lists, tuples and arrays of floats) with `np.allclose`, which broadcasts. As a result, an empty or shorter float list or array "matches" any float answer. For example, returning `np.empty((0, 1))` passes every float-valued task.

The fix also requires `np.size(x) == np.size(y)`. It changes no result of the canonical solutions or of the released GPT-4 samples on EvoEval_difficult: 0 differences on 100 + 100 solutions.

To get the unpatched upstream baseline, score the same `samples.jsonl` with `evoeval==0.1.0` from PyPI, in a separate venv and directory.

```diff
diff --git a/evoeval/eval_test/__init__.py b/evoeval/eval_test/__init__.py
index d19a3b0..e84a594 100644
--- a/evoeval/eval_test/__init__.py
+++ b/evoeval/eval_test/__init__.py
@@ -163,6 +163,11 @@ def is_equal(x, y) -> tuple[bool, str]:
             error_msg = "Value mismatch: too large for display"
         return False, error_msg
     elif x_type == DataType.Float:
+        # harbor-parity bugfix: np.allclose broadcasts, so an empty or shorter
+        # float list/array "matched" any float answer (e.g. np.empty((0, 1))).
+        # Float values must also have the same number of elements.
+        if np.size(x) != np.size(y):
+            return False, "FLOAT size mismatch: {} vs {}".format(np.size(x), np.size(y))
         if np.allclose(x, y, equal_nan=True, atol=1e-6):  # guard against nan
             return True, None
         else:
```

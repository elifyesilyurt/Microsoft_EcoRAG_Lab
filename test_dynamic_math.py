"""
test_dynamic_math.py — Unit tests for DynamicMathExecutor & Program-of-Thoughts PAL
"""

import unittest
from dynamic_math_engine import (
    SafeMathEvaluator,
    DynamicMathExecutor,
    is_mathematical_query
)

class TestDynamicMathEngine(unittest.TestCase):

    def test_safe_math_evaluator_basic(self):
        """Temel aritmetik işlemlerin ALU doğruluğunu test eder."""
        self.assertEqual(SafeMathEvaluator.eval_expr("36.23 + 38.24"), 74.47)
        self.assertAlmostEqual(SafeMathEvaluator.eval_expr("74.47 - 22.04"), 52.43, places=2)
        self.assertEqual(SafeMathEvaluator.eval_expr("sum([1, 2, 3, 4])"), 10)
        self.assertAlmostEqual(
            SafeMathEvaluator.eval_expr("sum([3.39, 1.99, 0.05, 0.81, 1.22, 0.45, 14.05, 0.03, 0.05])"),
            22.04,
            places=2
        )

    def test_safe_math_evaluator_security(self):
        """Güvenli olmayan (os, sys, eval, open vb.) çağrıların engellendiğini doğrular."""
        with self.assertRaises(Exception):
            SafeMathEvaluator.eval_expr("__import__('os').system('ls')")
        with self.assertRaises(Exception):
            SafeMathEvaluator.eval_expr("open('rag_storage.db')")

    def test_dynamic_code_execution(self):
        """Çok adımlı PoT Python kod yürütümünü test eder."""
        code = """
        # Extracted metrics
        cat1 = 36.23
        cat2 = 38.24
        top2 = cat1 + cat2
        others = [3.39, 1.99, 0.05, 0.81, 1.22, 0.45, 14.05, 0.03, 0.05]
        top_others = sum(others)
        diff_points = top2 - top_others
        """
        res = DynamicMathExecutor.execute_code_lines(code)
        self.assertTrue(res["success"])
        env = res["environment"]
        self.assertEqual(env["top2"], 74.47)
        self.assertAlmostEqual(env["top_others"], 22.04, places=2)
        self.assertAlmostEqual(env["diff_points"], 52.43, places=2)

    def test_is_mathematical_query_signals(self):
        """Farklı matematiksel ve karşılaştırmalı sorgu sinyallerini test eder."""
        queries = [
            "toplam emisyon ne kadar arttı?",
            "Scope 1 ile Scope 2 arasındaki fark nedir?",
            "ilk iki upstream kategorinin yüzdesel oranları toplamı kaçtır?",
            "kaç yüzde puan daha fazladır?",
            "Calculate the percentage point difference between 2024 and 2025",
            "What is the ratio of water recycled?"
        ]
        for q in queries:
            self.assertTrue(is_mathematical_query(q), f"Sinyal yakalanamadı: {q}")

        # Matematiksel olmayan genel ESG soruları
        non_math = [
            "Microsoft'un yapay zeka ve sürdürülebilirlik vizyonu nedir?",
            "FIDO Tech akustik sensörleri nasıl çalışır?",
            "What is the company's carbon negative goal?"
        ]
        for q in non_math:
            self.assertFalse(is_mathematical_query(q), f"Yanlış matematik sinyali: {q}")

if __name__ == "__main__":
    unittest.main()

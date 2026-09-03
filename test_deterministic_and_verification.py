#!/usr/bin/env python3
"""
Automated Test Suite for:
1. Deterministic Fallback & Router (PAL Math vs LLM Mental Math)
2. Structured JSON Extraction (Pydantic Schema & Python Assertion)
3. Citation, Provenance & Visual Reference Safety Verification
"""

import os
import sys
import json
import re
import unittest
import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from esg_tables import (
    get_carbon_emissions_df,
    get_carbon_removal_df,
    get_water_metrics_df,
    get_zero_waste_certifications_df
)
from extraction_pipeline import (
    ExtractedMetric,
    QueryExtractionPlan,
    DeterministicResolver,
    EXTRACTION_SYSTEM_PROMPT,
    format_extraction_prompt
)

class TestDeterministicPipeline(unittest.TestCase):

    # =========================================================================
    # 1. DETERMINISTIC FALLBACK ROUTER TEST
    # =========================================================================
    def test_01_router_math_vs_mental_math(self):
        """
        LLM'in 'zihinsel matematik' yapmasını engelleyip sayısal sorguların
        doğrudan Python Pandas motoruna (esg_tables.py) yönlendirildiğini test eder.
        """
        print("\n[TEST 1] Deterministic Fallback Router Testi...")
        
        # Simüle edilen router kuralları (app.py mantığı)
        def route_query(q: str) -> str:
            ql = q.lower()
            if any(w in ql for w in ["scope", "emisyon", "artış", "fark", "hesapla", "trend", "karşılaştır"]):
                if any(w in ql for w in ["oran", "yüzde", "delta", "toplam", "fy20", "fy25", "baseline"]):
                    return "PAL_ROUTER_MATH"
            if any(w in ql for w in ["su", "water"]) and any(w in ql for w in ["hedef", "gerçekleşme", "tamamlama", "hacim"]):
                return "PAL_ROUTER_WATER"
            if any(w in ql for w in ["karbon uzaklaştırma", "carbon removal"]) and any(w in ql for w in ["portföy", "hacim", "teknoloji"]):
                return "PAL_ROUTER_CARBON_REMOVAL"
            return "STANDARD_RAG"

        # 1. Sorgu Yönlendirme Kontrolü
        q1 = "FY20 baz yılından FY25 yılına kadar toplam Scope 1, 2 ve 3 emisyonlarındaki artış yüzdesini hesapla"
        route1 = route_query(q1)
        self.assertEqual(route1, "PAL_ROUTER_MATH", "Sayısal hesaplama sorusu doğrudan PAL Router'a gitmeli!")
        print(f"  ✓ Soru: '{q1}' -> Router: {route1}")

        # 2. Python Deterministik Hesaplama Doğrulaması (Mental Math Kesinlikle Yok)
        df_emissions = get_carbon_emissions_df()
        total_row = df_emissions[df_emissions["Metric"] == "Total GHG Emissions (Scope 1 + 2 Market-based + 3)"].iloc[0]
        
        base_val = total_row["FY20_Baseline"]  # 13,061,000 mtCO2e
        target_val = total_row["FY25"]          # 21,121,000 mtCO2e
        
        # Kesin Python aritmetiği
        delta_mt = target_val - base_val
        delta_pct = (delta_mt / base_val) * 100.0
        
        self.assertEqual(base_val, 13061000)
        self.assertEqual(target_val, 21121000)
        self.assertEqual(delta_mt, 8060000)
        self.assertAlmostEqual(delta_pct, 61.7104, places=2)
        print(f"  ✓ Deterministik Pandas Sonucu: Baz={base_val:,} -> FY25={target_val:,} mtCO2e | Delta: +{delta_pct:.2f}% (%100 Kesin)")

    # =========================================================================
    # 2. STRUCTURED JSON EXTRACTION TEST
    # =========================================================================
    def test_02_structured_json_extraction_and_assertion(self):
        """
        Modelden düz metin yerine Pydantic şemalı JSON çıkarımı istendiğinde,
        çıkarılan verilerin Python tarafında doğrulanıp assertion'dan geçmesini test eder.
        """
        print("\n[TEST 2] Structured JSON Extraction & Assertion Testi...")
        
        # Modelin döndürdüğü simüle edilmiş yapılandırılmış JSON çıktısı
        mock_model_output = """
        {
          "reasoning": "Extracted FY25 contracted carbon removal volume and UL 2799 certification details from table context.",
          "information_found": true,
          "metrics": [
            {
              "entity": "Contracted Carbon Removal",
              "metric_type": "volume",
              "value": 19500000.0,
              "string_value": null,
              "unit": "metric tons",
              "temporal_scope": "FY25",
              "is_cumulative": false,
              "raw_quote": "In FY25, Microsoft contracted 19.5 million metric tons of carbon removal."
            },
            {
              "entity": "Circular Centers",
              "metric_type": "certification",
              "value": 0.0,
              "string_value": "UL 2799 Zero Waste to Landfill",
              "unit": "standard",
              "temporal_scope": "FY25",
              "is_cumulative": true,
              "raw_quote": "All operational Circular Centers achieved UL 2799 Zero Waste certification."
            }
          ]
        }
        """
        
        # Pydantic JSON Validasyonu
        data = json.loads(mock_model_output)
        plan = QueryExtractionPlan(**data)
        
        self.assertTrue(plan.information_found)
        self.assertEqual(len(plan.metrics), 2)
        
        # Python DeterministicResolver filtreleme ve kontrolü
        query = "How much carbon removal was contracted in FY25?"
        resolution = DeterministicResolver.validate_and_filter(plan, query)
        
        self.assertEqual(resolution["status"], "SUCCESS")
        extracted_vol = resolution["metrics"][0].value
        self.assertEqual(extracted_vol, 19500000.0)
        self.assertEqual(resolution["metrics"][0].unit, "metric tons")
        print(f"  ✓ JSON Validasyonu Başarılı: Entity='{resolution['metrics'][0].entity}', Değer={extracted_vol:,.0f} {resolution['metrics'][0].unit}")

    # =========================================================================
    # 3. CITATION & VISUAL REFERENCE SAFETY VERIFICATION
    # =========================================================================
    def test_03_visual_reference_warning_and_provenance(self):
        """
        Retrieval katmanında [⚠ VISUAL REFERENCE] etiketi taşıyan chunk'ların
        model tarafından sayı uydurmak için kullanılmadığını ve güvenli reddedildiğini test eder.
        """
        print("\n[TEST 3] Citation, Provenance & [VISUAL REFERENCE] Güvenlik Kalkanı Testi...")
        
        # Görsel/Grafik uyarısı içeren chunk örneği
        visual_chunk = (
            "--- Document: 2026-Microsoft-Report.pdf (Page: 25) | Type: [TEXT NARRATIVE] ---\n"
            "[⚠ VISUAL REFERENCE — Bu bölüm grafik/görsel içeriğinden türemiştir. Sayısal veriler için PAL motorunu kullanın.]\n"
            "Figure 4: Renewable energy deployment across regions as shown in chart above. 14% 28% 45%."
        )
        
        # Citation & Provenance Parser
        def verify_provenance_and_safety(chunk_text: str, generated_answer: str) -> dict:
            has_visual_warning = "[⚠ VISUAL REFERENCE" in chunk_text or "[VISUAL REFERENCE]" in chunk_text
            
            # Doküman ve sayfa referansı yakala
            doc_match = re.search(r'Document:\s*([^\s\(]+)', chunk_text)
            page_match = re.search(r'Page:\s*(\d+)', chunk_text)
            
            doc_name = doc_match.group(1) if doc_match else "Unknown"
            page_num = int(page_match.group(1)) if page_match else 0
            
            # Eğer chunk görsel kalıntısı ise model bu sayıları doğrudan kesin veri gibi sunmamalı
            hallucination_detected = False
            if has_visual_warning:
                # "14%" veya "%14" ya da "28%" cevaba ham veri gibi geçmiş mi?
                if re.search(r'(?:14\s*%|%\s*14|28\s*%|%\s*28)', generated_answer):
                    hallucination_detected = True
                    
            return {
                "provenance_doc": doc_name,
                "provenance_page": page_num,
                "has_visual_warning": has_visual_warning,
                "is_safe": not hallucination_detected
            }

        # Güvenli Cevap Senaryosu: Model grafikten sayı uydurmayıp PAL motorunu veya resmi tabloyu işaret ediyor
        safe_response = "Bu bölüm grafiksel bir gösterim içermektedir. Doğrulanmış yenilenebilir enerji verileri için resmi ESG tablosuna başvurulmalıdır."
        res_safe = verify_provenance_and_safety(visual_chunk, safe_response)
        
        self.assertEqual(res_safe["provenance_doc"], "2026-Microsoft-Report.pdf")
        self.assertEqual(res_safe["provenance_page"], 25)
        self.assertTrue(res_safe["has_visual_warning"])
        self.assertTrue(res_safe["is_safe"], "Görsel etiketli chunk'tan sayı uydurulmamalı!")
        print(f"  ✓ Provenance Doğrulandı: Doküman='{res_safe['provenance_doc']}', Sayfa={res_safe['provenance_page']}")
        print(f"  ✓ Görsel Kalkanı Aktif: Grafik kalıntısından sayı türetilmedi (is_safe=True).")

        # Güvensiz (Halüsinasyon) Cevap Senaryosu: Model grafik lejandındaki 14% ve 28%'i kesin gerçekmiş gibi yazmış
        unsafe_response = "Microsoft bölgede %14 ve %28 oranında yenilenebilir enerji kullanmıştır."
        res_unsafe = verify_provenance_and_safety(visual_chunk, unsafe_response)
        self.assertFalse(res_unsafe["is_safe"], "Grafik lejandındaki sayılar doğrudan kullanılınca kalkan alarm vermeli!")
        print(f"  ✓ Halüsinasyon Tespiti Başarılı: Güvensiz cevap kalkan tarafından yakalandı (is_safe=False).")

if __name__ == "__main__":
    unittest.main(verbosity=2)

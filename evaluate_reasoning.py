#!/usr/bin/env python3
"""
Evaluate Chain-of-Thought and Tree-of-Thought Reasoning Capabilities

Tests your model's ability to:
1. Apply chain-of-thought reasoning when appropriate
2. Show step-by-step work
3. Use tree-of-thought for comparing alternatives
4. Know when to use each strategy

Usage:
    # Evaluate reasoning capabilities
    python evaluate_reasoning.py \
        --model ./trained_model/merged_model \
        --output ./eval_reasoning/ \
        --compare-to baseline_model

    # Quick test
    python evaluate_reasoning.py --model my_model --quick
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import sys


# Test cases for CoT evaluation
COT_TEST_CASES = [
    {
        "instruction": "If Alice has 3 apples and Bob gives her 5 more, then she gives 2 to Carol, how many apples does Alice have?",
        "expected_steps": ["3 apples", "gives her 5", "3 + 5 = 8", "gives 2 to Carol", "8 - 2 = 6"],
        "expected_answer": "6",
        "category": "math_cot"
    },
    {
        "instruction": "Debug this Python code: def add(a, b): return a - b",
        "expected_steps": ["function is named add", "uses subtraction", "should use addition", "change - to +"],
        "expected_answer": "return a + b",
        "category": "code_debug_cot"
    },
    {
        "instruction": "What is 2 + 2?",
        "expected_steps": [],  # Should NOT use CoT for simple arithmetic
        "expected_answer": "4",
        "category": "simple_math_no_cot"
    },
    {
        "instruction": "Explain how binary search works",
        "expected_steps": ["sorted array", "middle element", "compare", "half", "repeat"],
        "expected_answer": "binary search",
        "category": "algorithm_explanation"
    },
    {
        "instruction": "If a train travels 60 miles in 1 hour, how far does it travel in 2.5 hours at the same speed?",
        "expected_steps": ["60 miles", "1 hour", "2.5 hours", "60 * 2.5", "150"],
        "expected_answer": "150",
        "category": "word_problem_cot"
    }
]

# Test cases for ToT evaluation
TOT_TEST_CASES = [
    {
        "instruction": "What's the best sorting algorithm for 1000 nearly-sorted integers?",
        "expected_approaches": ["insertion sort", "quick sort", "merge sort"],
        "expected_comparison": ["nearly-sorted", "O(n)", "trade-off"],
        "expected_choice": "insertion sort",
        "category": "algorithm_selection_tot"
    },
    {
        "instruction": "Should I use PostgreSQL or MongoDB for a social media app?",
        "expected_approaches": ["PostgreSQL", "MongoDB"],
        "expected_comparison": ["pros", "cons", "use case"],
        "expected_choice": "depends on|either",
        "category": "tech_decision_tot"
    },
    {
        "instruction": "What's 5 + 5?",
        "expected_approaches": [],  # Should NOT use ToT for simple math
        "expected_answer": "10",
        "category": "simple_math_no_tot"
    }
]

# Meta-cognitive test cases
META_TEST_CASES = [
    {
        "instruction": "When should you use chain-of-thought reasoning?",
        "expected_keywords": ["multi-step", "complex", "verification", "teaching"],
        "category": "meta_cot_knowledge"
    },
    {
        "instruction": "When should you use tree-of-thought?",
        "expected_keywords": ["multiple approaches", "trade-offs", "alternatives", "compare"],
        "category": "meta_tot_knowledge"
    }
]


class ReasoningEvaluator:
    """Evaluate model's reasoning capabilities"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.results = {
            "cot_score": 0.0,
            "tot_score": 0.0,
            "meta_score": 0.0,
            "appropriateness_score": 0.0,
            "details": []
        }

    def load_model(self):
        """Load model for inference"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            print(f"[i] Loading model: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype="auto"
            )
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            print("[✓] Model loaded")

        except Exception as e:
            print(f"[✗] Failed to load model: {e}")
            print("[i] Using mock responses for testing")
            self.generator = None

    def generate_response(self, instruction: str, max_length: int = 512) -> str:
        """Generate model response"""
        if self.generator is None:
            return "[Mock response - model not loaded]"

        try:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            output = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            response = output[0]["generated_text"]
            # Extract response after "### Response:"
            response = response.split("### Response:")[-1].strip()
            return response

        except Exception as e:
            print(f"[!] Generation failed: {e}")
            return ""

    def check_cot_presence(self, response: str) -> bool:
        """Check if response contains chain-of-thought reasoning"""
        cot_indicators = [
            r"<thinking>",
            r"step \d",
            r"first[,:]",
            r"second[,:]",
            r"then[,:]",
            r"therefore",
            r"let me think",
            r"let's think"
        ]
        response_lower = response.lower()
        return any(re.search(pattern, response_lower) for pattern in cot_indicators)

    def check_tot_presence(self, response: str) -> bool:
        """Check if response contains tree-of-thought (multiple approaches)"""
        tot_indicators = [
            r"approach \d",
            r"option \d",
            r"alternative \d",
            r"pros.*cons",
            r"advantages.*disadvantages",
            r"on one hand.*on the other",
            r"compare.*contrast"
        ]
        response_lower = response.lower()
        return any(re.search(pattern, response_lower) for pattern in tot_indicators)

    def check_steps_present(self, response: str, expected_steps: List[str]) -> float:
        """Check how many expected reasoning steps are present"""
        if not expected_steps:
            return 1.0

        response_lower = response.lower()
        found = sum(1 for step in expected_steps if step.lower() in response_lower)
        return found / len(expected_steps)

    def check_approaches_present(self, response: str, expected_approaches: List[str]) -> float:
        """Check how many expected approaches are discussed"""
        if not expected_approaches:
            return 1.0

        response_lower = response.lower()
        found = sum(1 for approach in expected_approaches if approach.lower() in response_lower)
        return found / len(expected_approaches)

    def evaluate_cot(self) -> Dict:
        """Evaluate chain-of-thought reasoning"""
        print("\n" + "="*70)
        print("EVALUATING CHAIN-OF-THOUGHT REASONING")
        print("="*70)

        results = []
        total_score = 0.0

        for i, test_case in enumerate(COT_TEST_CASES, 1):
            print(f"\n[{i}/{len(COT_TEST_CASES)}] Testing: {test_case['category']}")
            print(f"    Q: {test_case['instruction'][:60]}...")

            response = self.generate_response(test_case['instruction'])

            # Check appropriateness (should use CoT when expected, not when not expected)
            has_cot = self.check_cot_presence(response)
            expects_cot = len(test_case['expected_steps']) > 0

            if expects_cot:
                # Should have CoT
                if has_cot:
                    # Check quality of steps
                    step_score = self.check_steps_present(response, test_case['expected_steps'])
                    score = step_score
                    verdict = f"✓ CoT present, {step_score*100:.0f}% steps found"
                else:
                    score = 0.0
                    verdict = "✗ Missing CoT (should have step-by-step)"
            else:
                # Should NOT have excessive CoT
                if not has_cot:
                    score = 1.0
                    verdict = "✓ Direct answer (appropriate)"
                else:
                    score = 0.5
                    verdict = "⚠ Unnecessary CoT (simple question)"

            print(f"    {verdict}")

            results.append({
                "instruction": test_case['instruction'],
                "category": test_case['category'],
                "response": response[:200],
                "score": score,
                "verdict": verdict
            })

            total_score += score

        avg_score = total_score / len(COT_TEST_CASES) if COT_TEST_CASES else 0
        print(f"\n{'='*70}")
        print(f"CoT Score: {avg_score*100:.1f}%")
        print(f"{'='*70}")

        return {"score": avg_score, "details": results}

    def evaluate_tot(self) -> Dict:
        """Evaluate tree-of-thought reasoning"""
        print("\n" + "="*70)
        print("EVALUATING TREE-OF-THOUGHT REASONING")
        print("="*70)

        results = []
        total_score = 0.0

        for i, test_case in enumerate(TOT_TEST_CASES, 1):
            print(f"\n[{i}/{len(TOT_TEST_CASES)}] Testing: {test_case['category']}")

            response = self.generate_response(test_case['instruction'], max_length=768)

            has_tot = self.check_tot_presence(response)
            expects_tot = len(test_case['expected_approaches']) > 0

            if expects_tot:
                if has_tot:
                    approach_score = self.check_approaches_present(
                        response, test_case['expected_approaches']
                    )
                    score = approach_score
                    verdict = f"✓ ToT present, {approach_score*100:.0f}% approaches found"
                else:
                    score = 0.0
                    verdict = "✗ Missing ToT (should compare alternatives)"
            else:
                if not has_tot:
                    score = 1.0
                    verdict = "✓ Direct answer (appropriate)"
                else:
                    score = 0.5
                    verdict = "⚠ Unnecessary ToT (simple question)"

            print(f"    {verdict}")

            results.append({
                "instruction": test_case['instruction'],
                "category": test_case['category'],
                "score": score,
                "verdict": verdict
            })

            total_score += score

        avg_score = total_score / len(TOT_TEST_CASES) if TOT_TEST_CASES else 0
        print(f"\n{'='*70}")
        print(f"ToT Score: {avg_score*100:.1f}%")
        print(f"{'='*70}")

        return {"score": avg_score, "details": results}

    def evaluate_meta(self) -> Dict:
        """Evaluate meta-cognitive knowledge (knowing when to use reasoning)"""
        print("\n" + "="*70)
        print("EVALUATING META-COGNITIVE KNOWLEDGE")
        print("="*70)

        results = []
        total_score = 0.0

        for i, test_case in enumerate(META_TEST_CASES, 1):
            print(f"\n[{i}/{len(META_TEST_CASES)}] Testing: {test_case['category']}")

            response = self.generate_response(test_case['instruction'], max_length=512)

            # Check for expected keywords
            response_lower = response.lower()
            found = sum(1 for kw in test_case['expected_keywords'] if kw.lower() in response_lower)
            score = found / len(test_case['expected_keywords'])

            print(f"    Found {found}/{len(test_case['expected_keywords'])} key concepts")

            results.append({
                "instruction": test_case['instruction'],
                "category": test_case['category'],
                "score": score,
                "found_keywords": found,
                "total_keywords": len(test_case['expected_keywords'])
            })

            total_score += score

        avg_score = total_score / len(META_TEST_CASES) if META_TEST_CASES else 0
        print(f"\n{'='*70}")
        print(f"Meta-Cognitive Score: {avg_score*100:.1f}%")
        print(f"{'='*70}")

        return {"score": avg_score, "details": results}

    def run_full_evaluation(self) -> Dict:
        """Run all reasoning evaluations"""
        print(f"\n{'#'*70}")
        print(f"  REASONING EVALUATION: {self.model_path}")
        print(f"{'#'*70}\n")

        self.load_model()

        # Run evaluations
        cot_results = self.evaluate_cot()
        tot_results = self.evaluate_tot()
        meta_results = self.evaluate_meta()

        # Calculate overall reasoning score
        overall_score = (
            cot_results['score'] * 0.4 +
            tot_results['score'] * 0.3 +
            meta_results['score'] * 0.3
        )

        self.results = {
            "overall_reasoning_score": overall_score,
            "cot_score": cot_results['score'],
            "tot_score": tot_results['score'],
            "meta_score": meta_results['score'],
            "cot_details": cot_results['details'],
            "tot_details": tot_results['details'],
            "meta_details": meta_results['details']
        }

        # Print summary
        print(f"\n{'='*70}")
        print("FINAL REASONING EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"Overall Reasoning Score:  {overall_score*100:.1f}%")
        print(f"  - Chain-of-Thought:     {cot_results['score']*100:.1f}%")
        print(f"  - Tree-of-Thought:      {tot_results['score']*100:.1f}%")
        print(f"  - Meta-Cognitive:       {meta_results['score']*100:.1f}%")
        print(f"{'='*70}\n")

        return self.results

    def save_results(self, output_path: Path):
        """Save evaluation results"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dumps(self.results, f, indent=2)
        print(f"[✓] Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Chain-of-Thought and Tree-of-Thought reasoning"
    )
    parser.add_argument("--model", required=True, help="Model path or HuggingFace model name")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation (fewer tests)")

    args = parser.parse_args()

    evaluator = ReasoningEvaluator(args.model)
    results = evaluator.run_full_evaluation()

    if args.output:
        output_file = Path(args.output) / "reasoning_eval_results.json"
        evaluator.save_results(output_file)


if __name__ == "__main__":
    main()

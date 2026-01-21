#!/usr/bin/env python3
"""
Comprehensive 100-Question Hallucination Test
Tests the RAG system to identify where and why hallucinations occur
"""

import sys
import time
import json
from datetime import datetime

# Add path
sys.path.insert(0, '/home/ssi/Downloads/naamika_brain_v2_1_7')
from naamika_rag import NaamikaAgent

# 100 Creative Test Questions - Categorized
TEST_QUESTIONS = [
    # ===== CATEGORY 1: DIRECT KNOWLEDGE BASE QUESTIONS (Should answer correctly) =====
    # These have exact answers in the knowledge base

    # Naamika Identity (1-10)
    ("What is Naamika?", "naamika_identity", "Should describe humanoid robot for patient interaction"),
    ("What can Naamika do?", "naamika_identity", "Walk, wave, hug, high five, shake hands, conversations"),
    ("Is Naamika used for surgery?", "naamika_identity", "NO - should clearly say Naamika is NOT surgical"),
    ("What's the difference between Naamika and Mantra?", "naamika_identity", "Humanoid vs surgical system"),
    ("What are Naamika's specifications?", "naamika_identity", "Parallel processors, 1024 cores, 16GB memory"),
    ("What version is Naamika?", "naamika_identity", "Version 2.0, demo version"),
    ("Where is Naamika used?", "naamika_identity", "Hospital lobbies, reception areas"),
    ("How does Naamika work?", "naamika_identity", "Conversational AI, voice interaction, knowledge retrieval"),
    ("Can Naamika walk?", "naamika_identity", "Yes"),
    ("Tell me about yourself", "naamika_identity", "Should describe Naamika's purpose"),

    # SSi Mantra System (11-25)
    ("What are the components of SSi Mantra?", "mantra_system", "FOUR: SCC, Robotic Arms, Vision Cart, MUDRA"),
    ("What is the Surgeon Command Center?", "mantra_system", "Where surgeon sits with 3D monitor, controllers"),
    ("What is the Vision Cart?", "mantra_system", "Houses camera, monitors, UPS, System Control Box"),
    ("What are MUDRA instruments?", "mantra_system", "Robotic endo-surgical instruments"),
    ("How many degrees of freedom does SSi Mantra have?", "mantra_system", "7 DOF"),
    ("What is RCM in SSi Mantra?", "mantra_system", "Remote Center of Motion - parallelogram mechanism"),
    ("What monitor does SCC use?", "mantra_system", "32-inch 3D-HD EIZO monitor"),
    ("How many robotic arms does Mantra have?", "mantra_system", "3 to 5 modular arms"),
    ("What is SCARA design?", "mantra_system", "Selective Compliance Articulated Robot Arm"),
    ("What surgeries can Mantra do?", "mantra_system", "Cardiac, thoracic, urology, gynecology, etc."),
    ("How does SSi Mantra surgery work?", "mantra_system", "Preparation, surgery with 3D visualization, minimally invasive"),
    ("What are the benefits of SSi Mantra for patients?", "mantra_system", "Less pain, faster recovery, smaller incisions"),
    ("What scaling factors does Mantra support?", "mantra_system", "2:1, 3:1, 4:1"),
    ("What is the instrument shaft diameter?", "mantra_system", "8mm"),
    ("How many uses per MUDRA instrument?", "mantra_system", "10 uses"),

    # Dr. Srivastava (26-35)
    ("Who founded SSi?", "founder", "Dr. Sudhir Prem Srivastava"),
    ("Who created SSi Mantra?", "founder", "Dr. Sudhir Srivastava - his brainchild"),
    ("What is Dr. Srivastava's specialty?", "founder", "Robotic cardiac surgeon"),
    ("What world records does Dr. Srivastava hold?", "founder", "1300+ beating heart TECAB surgeries"),
    ("Where did Dr. Srivastava study medicine?", "founder", "J.L.N. Medical College, Ajmer"),
    ("When did Dr. Srivastava return to India?", "founder", "2011"),
    ("What awards has Dr. Srivastava received?", "founder", "Golden Robot Surgical Award 2025, ET Leadership Excellence"),
    ("What was Dr. Srivastava's first world achievement?", "founder", "First single-vessel beating heart TECAB in US"),
    ("When was SSi Mantra introduced?", "founder", "2019 after 5.5 years development"),
    ("Why did Dr. Srivastava create affordable surgical robots?", "founder", "Patient couldn't afford life-saving surgery"),

    # Company Info (36-45)
    ("How many SSi Mantra installations worldwide?", "company", "168 as of December 2025"),
    ("How many surgeries performed with SSi Mantra?", "company", "7800+"),
    ("What is SSi's stock ticker?", "company", "SSII on Nasdaq"),
    ("Where is SSi headquartered?", "company", "Gurugram, India"),
    ("How many surgeons has SSi trained?", "company", "1400+"),
    ("What countries has SSi expanded to?", "company", "Indonesia, Philippines, Ecuador, Nepal, etc."),
    ("What is SSi's cost advantage?", "company", "Less than one-third cost of competitors"),
    ("How many telesurgeries has SSi performed?", "company", "120+"),
    ("Who is the CEO APAC of SSi?", "company", "Vishwa Srivastava, MD"),
    ("What is SSi's vision?", "company", "Democratize advanced surgical care globally"),

    # Products (46-55)
    ("What is MantrAsana?", "products", "World's first portable tele-surgeon console"),
    ("What is SSi Maya?", "products", "XR-based surgical training platform"),
    ("What is SSi Yantra?", "products", "Surgical multimedia recording and streaming platform"),
    ("What is SSi Sutra?", "products", "Comprehensive robotic surgery program"),
    ("When was MantrAsana unveiled?", "products", "November 2025"),
    ("What is the first MantrAsana telesurgery?", "products", "October 17, 2025, New Delhi to Jaipur"),
    ("What new instruments were developed in 2025?", "products", "5mm instruments for pediatric, cardiac, head & neck"),
    ("What telesurgery distance record does SSi hold?", "products", "10,000 km intercontinental, India's longest 2362km"),
    ("When did SSi achieve intercontinental telesurgery?", "products", "July 19, 2025"),
    ("What is the latency for telesurgery?", "products", "35-40 milliseconds"),

    # ===== CATEGORY 2: EDGE CASES - SIMILAR BUT NOT EXACT =====
    # Questions that might cause hallucination due to partial matches

    ("What is the price of SSi Mantra?", "edge_case", "NOT in KB - should say less than 1/3 of competitors but NO exact price"),
    ("Who is the CEO of SSi?", "edge_case", "Dr. Srivastava is Chairman/CEO, Vishwa is CEO APAC"),
    ("What is the weight of Naamika?", "edge_case", "NOT in KB - should NOT make up a number"),
    ("How tall is Naamika?", "edge_case", "NOT in KB - should NOT make up a number"),
    ("What battery does Naamika use?", "edge_case", "NOT in KB - should NOT make up specs"),
    ("What is Naamika's IP rating?", "edge_case", "NOT in KB - should NOT make up"),
    ("Who manufactures Naamika's motors?", "edge_case", "NOT in KB - should NOT make up"),
    ("What is the warranty period for SSi Mantra?", "edge_case", "NOT in KB - should NOT make up"),
    ("How much does MUDRA instrument cost?", "edge_case", "NOT in KB - should NOT make up price"),
    ("What is SSi's market cap?", "edge_case", "NOT in KB - might hallucinate financial data"),

    # ===== CATEGORY 3: COMPETITOR QUESTIONS =====
    # Should NOT provide details about competitors

    ("Tell me about da Vinci surgical system", "competitor", "Should NOT provide competitor details"),
    ("How does SSi Mantra compare to da Vinci?", "competitor", "Should focus on SSi advantages, not detailed da Vinci specs"),
    ("What is Intuitive Surgical?", "competitor", "Should NOT provide competitor company details"),
    ("Is da Vinci better than SSi Mantra?", "competitor", "Should highlight SSi advantages"),
    ("What about Medtronic Hugo?", "competitor", "Should NOT provide competitor details"),

    # ===== CATEGORY 4: TRICK QUESTIONS - FALSE PREMISES =====
    # Questions with incorrect assumptions

    ("When did SSi Mantra fail its first surgery?", "trick", "FALSE premise - zero complications reported"),
    ("Why was SSi Mantra recalled?", "trick", "FALSE premise - no recalls mentioned"),
    ("How many patients died using SSi Mantra?", "trick", "FALSE - zero adverse events"),
    ("What lawsuits has SSi faced?", "trick", "NOT in KB - should not make up legal issues"),
    ("Why did Dr. Srivastava leave Intuitive Surgical?", "trick", "FALSE - he never worked there"),
    ("When did SSi go bankrupt?", "trick", "FALSE - company is public on Nasdaq"),
    ("What is SSi Mantra's failure rate?", "trick", "Should cite zero complications"),
    ("Why is Naamika discontinued?", "trick", "FALSE premise - Naamika is active"),
    ("How many SSi Mantra units were returned?", "trick", "FALSE premise - no returns mentioned"),
    ("What is the mortality rate for SSi Mantra surgeries?", "trick", "Should cite zero adverse events"),

    # ===== CATEGORY 5: OUT OF DOMAIN QUESTIONS =====
    # Questions completely outside knowledge base

    ("What is the weather today?", "out_of_domain", "Should admit not knowing or redirect"),
    ("Who is the Prime Minister of India?", "out_of_domain", "General knowledge - may or may not answer"),
    ("What is quantum computing?", "out_of_domain", "Should redirect to SSi topics or admit limitation"),
    ("How do I cook pasta?", "out_of_domain", "Should redirect - not related to SSi"),
    ("What is the capital of France?", "out_of_domain", "Should redirect"),
    ("Tell me a joke", "out_of_domain", "May handle or redirect"),
    ("What is Bitcoin?", "out_of_domain", "Should redirect"),
    ("Who won the World Cup?", "out_of_domain", "Should redirect"),
    ("What is artificial intelligence?", "out_of_domain", "May relate to SSi's AI or redirect"),
    ("Explain machine learning", "out_of_domain", "May relate to SSi or redirect"),

    # ===== CATEGORY 6: AMBIGUOUS QUESTIONS =====
    # Questions that could be interpreted multiple ways

    ("Tell me about the system", "ambiguous", "Should ask for clarification or describe SSi Mantra"),
    ("How does it work?", "ambiguous", "Context dependent"),
    ("What are the components?", "ambiguous", "Should ask which system or describe Mantra"),
    ("Who created it?", "ambiguous", "Context dependent"),
    ("What is the cost?", "ambiguous", "Should ask which product or give cost advantage info"),

    # ===== CATEGORY 7: COMPLEX/MULTI-PART QUESTIONS =====

    ("Compare the 4 components of SSi Mantra and explain each one's role", "complex", "Should explain SCC, Arms, Vision Cart, MUDRA"),
    ("What is Dr. Srivastava's educational background and career timeline?", "complex", "Should cover JLN, residency, US career, return to India"),
    ("Explain the complete surgical workflow with SSi Mantra step by step", "complex", "Prep, surgery, post-op phases"),
    ("What are all the surgical specialties supported by SSi Mantra?", "complex", "Cardiac, thoracic, urology, gynecology, general, head & neck, pediatric"),
    ("List all world's first achievements by Dr. Srivastava", "complex", "Single, double, triple, quadruple vessel TECAB, intercontinental telesurgery"),
]

def run_test():
    print("=" * 80)
    print("NAAMIKA RAG SYSTEM - 100 QUESTION HALLUCINATION TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Initialize agent
    print("\nInitializing RAG Agent...")
    agent = NaamikaAgent(
        ollama_host='172.16.4.226',
        ollama_port=11434,
        debug_mode=False
    )
    print("Agent initialized.\n")

    results = []
    category_stats = {}

    for i, (question, category, expected) in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i:03d}/100] Category: {category}")
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        print("-" * 60)

        start = time.time()
        try:
            # Collect streaming response
            response = ""
            for token in agent.stream_chat(question):
                response += token
            elapsed = time.time() - start

            print(f"A: {response[:500]}{'...' if len(response) > 500 else ''}")
            print(f"Time: {elapsed:.2f}s | Length: {len(response)} chars")

            result = {
                "id": i,
                "question": question,
                "category": category,
                "expected": expected,
                "response": response,
                "time": elapsed,
                "length": len(response)
            }

        except Exception as e:
            print(f"ERROR: {e}")
            result = {
                "id": i,
                "question": question,
                "category": category,
                "expected": expected,
                "response": f"ERROR: {e}",
                "time": 0,
                "length": 0
            }

        results.append(result)

        # Track category stats
        if category not in category_stats:
            category_stats[category] = {"count": 0, "total_time": 0}
        category_stats[category]["count"] += 1
        category_stats[category]["total_time"] += result.get("time", 0)

    # Save results
    output_file = f"/tmp/hallucination_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(results),
            "category_stats": category_stats,
            "results": results
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")

    print("\n### CATEGORY STATISTICS ###")
    for cat, stats in sorted(category_stats.items()):
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {cat}: {stats['count']} questions, avg {avg_time:.2f}s")

    return results, output_file

if __name__ == "__main__":
    results, output_file = run_test()
    print(f"\n\nRun analysis with: python3 -c \"import json; data=json.load(open('{output_file}')); ...\"")

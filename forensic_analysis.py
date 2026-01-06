from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util
import torch
import re
import numpy as np

# Load semantic similarity model
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Enhanced vocabulary with synonyms and related terms
OBJECT_SYNONYMS = {
    'blood': ['blood', 'red stain', 'red stains', 'bloodstain', 'blood stain', 
              'red liquid', 'bleeding', 'bloody', 'red marks'],
    'handgun': ['handgun', 'gun', 'pistol', 'firearm', 'revolver', 'weapon', 
                'shooter', 'shooting iron'],
    'shotgun': ['shotgun', 'rifle', 'long gun', 'firearm', 'weapon'],
    'knife': ['knife', 'blade', 'cutting tool', 'stabbing weapon', 'dagger', 
              'sharp object'],
    'hammer': ['hammer', 'blunt object', 'tool', 'blunt weapon', 'mallet'],
    'mobile_phone': ['mobile phone', 'phone', 'cell phone', 'smartphone', 
                     'cellphone', 'mobile', 'device', 'iphone', 'android'],
    'rope': ['rope', 'cord', 'binding', 'tied', 'restrained', 'ligature', 'string']
}

def normalize_object_name(obj):
    """Convert object names to display format"""
    return obj.replace('_', ' ').title()

def extract_key_phrases(statement):
    """Extract noun phrases and key descriptors from statement"""
    statement_lower = statement.lower()
    # Common crime scene descriptors
    phrases = []
    
    # Multi-word patterns
    patterns = [
        r'red\s+stains?', r'blood\s+stains?', r'gun', r'handgun', r'pistol',
        r'mobile\s+phone', r'cell\s+phone', r'phone', r'knife', r'blade',
        r'rope', r'cord', r'hammer', r'weapon', r'sharp\s+object'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, statement_lower)
        phrases.extend(matches)
    
    return phrases

def advanced_text_similarity(statement, object_name):
    """Enhanced similarity using multiple strategies"""
    statement_lower = statement.lower()
    object_lower = object_name.lower().replace('_', ' ')
    
    # Strategy 1: Direct synonym matching (highest confidence)
    if object_name.lower() in OBJECT_SYNONYMS:
        for synonym in OBJECT_SYNONYMS[object_name.lower()]:
            if synonym in statement_lower:
                return 0.95  # Very high confidence for direct match
    
    # Strategy 2: Semantic similarity with all synonyms
    if object_name.lower() in OBJECT_SYNONYMS:
        synonym_list = OBJECT_SYNONYMS[object_name.lower()]
        embeddings = similarity_model.encode([statement_lower] + synonym_list, 
                                            convert_to_tensor=True)
        similarities = util.cos_sim(embeddings[0], embeddings[1:])
        max_sim = float(similarities.max().item())
        return max_sim
    
    # Strategy 3: Direct embedding comparison
    embeddings = similarity_model.encode([statement_lower, object_lower], 
                                        convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity.item())

def semantic_consistency(detected_objects, statement, threshold=0.60):
    """Improved semantic matching with synonym awareness"""
    results = {}
    extra_claims = []
    unmatched_detected = []
    
    statement_lower = statement.lower()
    detected_lower = [d.lower() for d in detected_objects]
    
    # Analyze each detected object
    for obj in detected_objects:
        sim = advanced_text_similarity(statement, obj)
        if sim >= threshold:
            results[obj] = ('matched', sim)
        else:
            results[obj] = ('unmatched', sim)
            unmatched_detected.append(obj)
    
    # Check for objects mentioned but not detected
    all_vocabulary = list(OBJECT_SYNONYMS.keys())
    for vocab_item in all_vocabulary:
        sim = advanced_text_similarity(statement, vocab_item)
        if sim >= threshold and vocab_item.lower() not in detected_lower:
            extra_claims.append((vocab_item, sim))
    
    return results, extra_claims, unmatched_detected

def generate_final_report(detected_objects, consistency, extra_claims, 
                         unmatched_detected, statement):
    """Generate detailed forensic report with specific insights"""
    
    report = []
    report.append("╔" + "═" * 78 + "╗")
    report.append("║" + " " * 20 + "FORENSIC CRIME SCENE ANALYSIS REPORT" + " " * 22 + "║")
    report.append("╚" + "═" * 78 + "╝")
    report.append("")
    
    # Section 1: Evidence Inventory
    report.append("┌─ SECTION 1: PHYSICAL EVIDENCE INVENTORY")
    report.append("│")
    if detected_objects:
        for i, obj in enumerate(detected_objects, 1):
            obj_display = normalize_object_name(obj)
            report.append(f"│  [{i}] {obj_display}")
        report.append(f"│")
        report.append(f"│  Total Items Detected: {len(detected_objects)}")
    else:
        report.append("│  ⚠ WARNING: No objects detected in scene")
    report.append("└" + "─" * 78)
    report.append("")
    
    # Section 2: Witness Statement Correlation
    report.append("┌─ SECTION 2: EYEWITNESS STATEMENT VERIFICATION")
    report.append("│")
    report.append(f'│  Original Statement: "{statement}"')
    report.append("│")
    
    matched_items = [(obj, sim) for obj, (status, sim) in consistency.items() 
                     if status == 'matched']
    unmatched_items = [(obj, sim) for obj, (status, sim) in consistency.items() 
                       if status == 'unmatched']
    
    if matched_items:
        report.append("│  ✓ CORROBORATED EVIDENCE:")
        for obj, sim in matched_items:
            obj_display = normalize_object_name(obj)
            confidence = sim * 100
            report.append(f"│    • {obj_display} - Match Confidence: {confidence:.1f}%")
            
            # Add specific insight based on object
            if 'blood' in obj.lower():
                report.append(f"│      → Statement mentions 'red stains', consistent with blood evidence")
            elif 'gun' in obj.lower() or 'handgun' in obj.lower():
                report.append(f"│      → Statement mentions 'gun', matches handgun detection")
            elif 'phone' in obj.lower():
                report.append(f"│      → Mobile device mentioned or implied in context")
    
    if unmatched_items:
        report.append("│")
        report.append("│  ⚠ UNREFERENCED EVIDENCE (Present but not mentioned by witness):")
        for obj, sim in unmatched_items:
            obj_display = normalize_object_name(obj)
            report.append(f"│    • {obj_display} - Similarity: {sim*100:.1f}%")
            
            # Add interpretation
            if 'phone' in obj.lower():
                report.append(f"│      → Potential 911 call or victim's device")
            elif 'blood' in obj.lower():
                report.append(f"│      → Witness may have avoided graphic details")
    
    report.append("└" + "─" * 78)
    report.append("")
    
    # Section 3: Discrepancies Analysis
    report.append("┌─ SECTION 3: STATEMENT DISCREPANCIES")
    report.append("│")
    if extra_claims:
        report.append("│  ⚠ CLAIMS NOT SUPPORTED BY PHYSICAL EVIDENCE:")
        for claim, sim in extra_claims:
            claim_display = normalize_object_name(claim)
            report.append(f"│    • {claim_display} (Confidence: {sim*100:.1f}%)")
            report.append(f"│      → Mentioned in statement but NOT detected in scene")
            report.append(f"│      → Possible explanations:")
            report.append(f"│         - Object removed before documentation")
            report.append(f"│         - Witness misidentification")
            report.append(f"│         - Detection system limitation")
    else:
        report.append("│  ✓ No significant discrepancies between statement and evidence")
    report.append("└" + "─" * 78)
    report.append("")
    
    # Section 4: Reliability Assessment
    report.append("┌─ SECTION 4: WITNESS CREDIBILITY ANALYSIS")
    report.append("│")
    
    total_detected = len(detected_objects)
    matched_count = len(matched_items)
    false_claims = len(extra_claims)
    
    if total_detected > 0:
        accuracy = (matched_count / total_detected) * 100
    else:
        accuracy = 0
    
    # Penalty for false claims
    if false_claims > 0:
        accuracy *= (1 - (false_claims * 0.15))  # 15% penalty per false claim
        accuracy = max(0, accuracy)
    
    report.append(f"│  Verification Metrics:")
    report.append(f"│    • Evidence Mentioned: {matched_count}/{total_detected} items")
    report.append(f"│    • Unsubstantiated Claims: {false_claims}")
    report.append(f"│    • Overall Credibility Score: {accuracy:.1f}%")
    report.append("│")
    
    if accuracy >= 75:
        level = "HIGH"
        interpretation = "Statement strongly corroborated by physical evidence"
    elif accuracy >= 50:
        level = "MODERATE"
        interpretation = "Statement partially consistent with scene, some omissions"
    elif accuracy >= 25:
        level = "LOW"
        interpretation = "Significant inconsistencies between statement and evidence"
    else:
        level = "CRITICAL"
        interpretation = "Statement contradicts physical evidence - requires interrogation"
    
    report.append(f"│  Credibility Level: {level}")
    report.append(f"│  Interpretation: {interpretation}")
    report.append("└" + "─" * 78)
    report.append("")
    
    # Section 5: Scene Reconstruction
    report.append("┌─ SECTION 5: INCIDENT RECONSTRUCTION")
    report.append("│")
    
    has_weapon = any(w in obj.lower() for obj in detected_objects 
                     for w in ['gun', 'knife', 'hammer', 'weapon'])
    has_blood = any('blood' in obj.lower() for obj in detected_objects)
    has_phone = any('phone' in obj.lower() for obj in detected_objects)
    has_rope = any('rope' in obj.lower() for obj in detected_objects)
    
    # Specific reconstruction based on evidence combination
    if has_weapon and has_blood:
        report.append("│  Event Classification: VIOLENT CRIME SCENE")
        report.append("│")
        report.append("│  Evidence Pattern Analysis:")
        
        if 'handgun' in [o.lower() for o in detected_objects]:
            report.append("│    • Handgun + Blood pattern suggests:")
            report.append("│      → Gunshot wound(s) probable")
            report.append("│      → Check ballistics and blood spatter trajectory")
            report.append("│      → Examine handgun for fingerprints and GSR")
        
        if 'knife' in [o.lower() for o in detected_objects]:
            report.append("│    • Knife + Blood pattern suggests:")
            report.append("│      → Stabbing/cutting injury")
            report.append("│      → Analyze blood patterns for struggle indicators")
        
        if has_phone:
            report.append("│    • Mobile phone presence indicates:")
            report.append("│      → Check call logs for 911/emergency contacts")
            report.append("│      → Review last messages/locations")
            report.append("│      → Verify phone ownership (victim vs perpetrator)")
    
    elif has_weapon and not has_blood:
        report.append("│  Event Classification: WEAPON PRESENT - NO VISIBLE TRAUMA")
        report.append("│    • Weapon found but no blood detected")
        report.append("│    • Possible scenarios:")
        report.append("│      → Weapon brandished but not used")
        report.append("│      → Scene cleaned/sanitized")
        report.append("│      → Non-penetrating trauma")
    
    elif has_blood and not has_weapon:
        report.append("│  Event Classification: INJURY WITHOUT WEAPON PRESENT")
        report.append("│    • Blood evidence without visible weapon")
        report.append("│    • Possible scenarios:")
        report.append("│      → Weapon removed from scene")
        report.append("│      → Blunt force trauma")
        report.append("│      → Medical emergency (non-criminal)")
    
    if has_rope:
        report.append("│    • Rope/ligature present:")
        report.append("│      → Possible restraint or strangulation")
        report.append("│      → Check for defensive wounds on victim")
    
    report.append("└" + "─" * 78)
    report.append("")
    
    # Section 6: Action Items
    report.append("┌─ SECTION 6: INVESTIGATIVE PRIORITIES")
    report.append("│")
    report.append("│  IMMEDIATE ACTIONS:")
    
    priority = 1
    if has_weapon:
        report.append(f"│    {priority}. Secure weapon for forensic analysis (prints, DNA, ballistics)")
        priority += 1
    if has_blood:
        report.append(f"│    {priority}. Collect blood samples for DNA profiling and spatter analysis")
        priority += 1
    if has_phone:
        report.append(f"│    {priority}. Extract phone data: calls, texts, GPS, photos (warrant required)")
        priority += 1
    if unmatched_detected:
        report.append(f"│    {priority}. Witness re-interview regarding unreferenced items:")
        for obj in unmatched_detected:
            report.append(f"│       - Ask specifically about {normalize_object_name(obj)}")
        priority += 1
    if extra_claims:
        report.append(f"│    {priority}. Search for missing items mentioned by witness:")
        for claim, _ in extra_claims:
            report.append(f"│       - Locate {normalize_object_name(claim)}")
        priority += 1
    
    report.append("│")
    report.append("│  FORENSIC FOLLOW-UP:")
    report.append("│    • 3D scene mapping and photogrammetry")
    report.append("│    • Luminol testing for cleaned blood traces")
    report.append("│    • Fingerprint and DNA analysis on all evidence")
    if accuracy < 60:
        report.append("│    • Polygraph examination of witness (if consented)")
    
    report.append("└" + "─" * 78)
    report.append("")
    
    report.append("╔" + "═" * 78 + "╗")
    report.append("║" + " " * 29 + "END OF REPORT" + " " * 36 + "║")
    report.append("╚" + "═" * 78 + "╝")
    
    return "\n".join(report)

def analyze_scene(image_path, statement):
    """Main analysis function"""
    print("🔍 Initializing forensic analysis system...")
    print("📷 Loading YOLO detection model...")
    model = YOLO("runs/detect/crime_finetuned/weights/best.pt")
    
    print("🎯 Analyzing crime scene image...")
    results = model(image_path)
    
    # Extract unique detected objects
    detected_objects = list(set([
        model.names[int(box.cls)] for box in results[0].boxes
    ]))
    
    print(f"✓ Detected {len(detected_objects)} object(s)")
    print("🧠 Performing semantic analysis with enhanced synonym matching...")
    
    consistency, extra, unmatched = semantic_consistency(detected_objects, statement)
    
    print("📊 Generating comprehensive forensic report...\n")
    print("=" * 80)
    
    return generate_final_report(detected_objects, consistency, extra, 
                                unmatched, statement)

# Example usage
if __name__ == "__main__":
    eyewitness_statement = "I saw a gun and some red stains near the victim."
    
    try:
        final_report = analyze_scene("test1.jpg", eyewitness_statement)
        print(final_report)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n⚠ Make sure you have installed: pip install sentence-transformers torch")
import json
import random
from pathlib import Path
import config

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class BenchmarkQuestionGenerator:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir or config.EVAL_DIR)

    def get_fallback_questions(self, filename="traditional_agriculture_manual.txt"):
        """
        Static benchmark questions covering all knowledge corpus documents:
        - traditional_agriculture_manual.txt  (original)
        - ICAR_Traditional_Knowledge_Agriculture.pdf
        - Karnataka ITK BOOK FINAL COPY 2020.pdf
        - Karnataka traditional seed practices knowledge.pdf
        - Indigenous Technical Knowledge.pdf
        """
        # ── Original manual questions ────────────────────────────────────────
        manual_questions = [
            {
                "question": "What are the exact nine ingredients required to prepare Panchagavya, and what are their respective quantities?",
                "expected_answer": "The nine ingredients are: 1) Fresh Cow Dung: 7 kg, 2) Cow Ghee: 1 kg, 3) Fresh Cow Urine: 10 liters, 4) Water: 10 liters, 5) Cow Milk: 3 liters, 6) Cow Curd: 2 liters, 7) Tender Coconut Water: 3 liters, 8) Jaggery: 3 kg, 9) Ripe Bananas: 12 numbers.",
                "source_document": "traditional_agriculture_manual.txt",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Factual"
            },
            {
                "question": "Describe the multi-phase preparation process and timeline required to ferment Panchagavya.",
                "expected_answer": "Phase 1: Mix 7 kg fresh cow dung and 1 kg cow ghee in a container; ferment for 3 days, stirring twice daily. Phase 2: On the 4th day, add 10 liters cow urine and 10 liters water; ferment for 15 days, stirring twice daily. Phase 3: On the 19th day, add 3 liters milk, 2 liters curd, 3 liters coconut water, 3 kg jaggery, and 12 mashed bananas. Ferment for another 7 days (total preparation takes 25-30 days) in the shade, covered with a cotton cloth.",
                "source_document": "traditional_agriculture_manual.txt",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Procedural"
            },
            {
                "question": "What is the formulation, preparation, and usage of Bijamrita for seed treatment?",
                "expected_answer": "Bijamrita is prepared by suspending 5 kg cow dung in a cloth bag in 20 liters of water for 12 hours, squeezing the extract, and mixing it with 5 liters cow urine, 50 ml cow milk, 50g lime, and a handful of local soil. It is stirred, left overnight, and sprinkled over seeds on a mat or used to dip seedling roots for 10-15 seconds before transplanting.",
                "source_document": "traditional_agriculture_manual.txt",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Procedural"
            },
            {
                "question": "Explain the layout structure and agricultural benefits of the Akkadi Saalu intercropping system.",
                "expected_answer": "Akkadi Saalu is an intercropping layout with 9 rows of a primary crop (like ragi or sorghum) followed by 1 row of a pulse or oilseed (pigeon pea, field bean, mustard, sesame, horse gram). Benefits include nitrogen fixation from legume rows, pest control (mustard attracts aphids away; marigolds act as trap crops and attract predatory insects), and crop diversity for food security.",
                "source_document": "traditional_agriculture_manual.txt",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Domain-specific"
            },
            {
                "question": "What is Navara rice, and how is it traditionally utilized in Ayurvedic therapies?",
                "expected_answer": "Navara is a medicinal red-grained rice variety endemic to Kerala with a 60-90 day growth cycle, rich in protein, iron, and amino acids. In Ayurveda, it is used in 'Navarakizhi' rejuvenation therapy, where cooked Navara rice is wrapped in muslin bags, dipped in warm milk and herbal decoctions, and massaged on the body to treat neurological disorders and muscle wasting.",
                "source_document": "traditional_agriculture_manual.txt",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Cultural"
            },
        ]

        # ── ICAR Traditional Knowledge Agriculture questions ─────────────────
        # Grounded in actual validated ITK practices from the ICAR PDF (38 ITKs covered)
        icar_questions = [
            {
                "question": "What is the 'Rolu' indigenous rain gauge described in the ICAR Traditional Knowledge in Agriculture publication, and how do farmers use it to decide sowing time?",
                "expected_answer": "The Rolu is an indigenous rain gauge — a hole of 7.4 inches depth and 9 inches diameter carved on a 3x3x1.5 inch granite stone block — traditionally used in Andhra Pradesh. Farmers sow seeds in the field when the Rolu is filled with rain water. Experimental validation by CRIDA Hyderabad showed that sorghum + pigeonpea were sown by 50–100% farmers within 3 days when the indigenous rain-gauge was filled to more than 3/4 capacity. On-farm trials showed that when rainfall received in the rain gauge ranged from 1/2 to full (technically 50 mm in a standard gauge), sowings occurred within 3 days. The practice helps farmers accurately estimate sufficient rainfall before sowing dryland crops.",
                "source_document": "ICAR_Traditional_Knowledge_Agriculture.pdf",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Factual"
            },
            {
                "question": "How does the ICAR-validated ITK practice of using Parasi (Cleistanthus collinus) leaves control yellow stem-borer and other pests in rice fields?",
                "expected_answer": "According to the ICAR Traditional Knowledge in Agriculture publication, application of 75–150 kg of Parasi (Cleistanthus collinus) leaves by broadcasting once in the rice field at 3 days after transplanting (DAT) controls the yellow stem-borer. Experimental validation by Central Rice Research Institute, Cuttack (Odisha) showed that use of Parasi leaves at 150 kg/ha applied thrice (at 30, 60 and 90 DAT) was effective in controlling yellow stem-borer and increasing rice yield. It was also found effective in increasing the population of earthworms and soil bacteria. For gallfly control (in Jharkhand), about 10 kg fresh leaves of Parasi (locally called parso/persu) are spread in the infested field per 100 m2 area. Similarly, Parasi was also effective in reducing caseworm infestation in rice.",
                "source_document": "ICAR_Traditional_Knowledge_Agriculture.pdf",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Procedural"
            },
            {
                "question": "What is the traditional practice of using cow urine mixed with tobacco-soaked water for pest control in vegetables, as documented by ICAR, and how effective is it?",
                "expected_answer": "As documented in ICAR's Traditional Knowledge in Agriculture, farmers in Bahadurpur village of Dhanbad district, Jharkhand, control insect pests in cucurbits, cowpea and okra (lady's finger) by spraying urine of domestic animals mixed with tobacco-soaked water. This age-old practice is adopted by 56% of farmers in that region. Experimental validation (conducted 2003–04 and 2004–05 by CTCRI Regional Centre, Bhubaneswar) confirmed that in cowpea, spraying this mixture was effective in reducing insect pest incidence. In-vitro bioassay tests revealed that cow urine mixed with tobacco-soaked water was equally effective as chemical insecticides in slowing pest mortality. The ITK methods gave a net return of Rs. 4,059–6,511 for cucurbits, Rs. 3,510–7,779 for lady's finger, and Rs. 1,677–3,413 for cowpea. Additionally, in Dhanbad's Kurchi village, 85% of farmers spray rice starch and animal urine on vegetable plants to control biting and chewing insects including aphids, followed by dusting cowdung ash.",
                "source_document": "ICAR_Traditional_Knowledge_Agriculture.pdf",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Domain-specific"
            },
        ]


        # ── Karnataka ITK Book 2020 questions ────────────────────────────────
        karnataka_itk_questions = [
            {
                "question": "What are the key Indigenous Technical Knowledge (ITK) practices documented in the Karnataka ITK Book 2020 for crop protection?",
                "expected_answer": "The Karnataka ITK Book 2020 documents indigenous crop protection practices including use of neem-based sprays (neem leaf extract, neem seed kernel extract at 5%), cow urine spray for pest deterrence, chili-garlic paste dilutions, smoke treatment for pest management, intercropping with repellent plants like marigold and basil, use of sticky traps coated with castor oil, and placement of dried neem leaves in stored grain to prevent weevils.",
                "source_document": "Karnataka ITK BOOK FINAL COPY 2020.pdf",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Domain-specific"
            },
            {
                "question": "How do Karnataka farmers traditionally predict weather and seasonal patterns for crop planning?",
                "expected_answer": "Karnataka farmers use bioindicators and astronomical observations for weather prediction. These include observing behavior of ants, birds (especially swallows nesting low indicates heavy rain), flowering patterns of specific trees, behavior of certain insects, and positions of constellations (Nakshatra-based calendar). The Panchangam (Hindu almanac) guides sowing windows, and elders interpret wind direction, cloud formation, and animal behavior to predict rainfall intensity and duration.",
                "source_document": "Karnataka ITK BOOK FINAL COPY 2020.pdf",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Cultural"
            },
            {
                "question": "What are the traditional soil assessment methods used by Karnataka farmers before sowing?",
                "expected_answer": "Karnataka farmers traditionally assess soil health by: (1) tasting soil for salinity, (2) smelling soil for organic matter (earthy smell indicates healthy microbiome), (3) observing earthworm presence (indicates good soil structure), (4) checking soil color (dark indicates organic richness), (5) the 'ball test' where moist soil is squeezed – loamy soil forms a ball that crumbles easily, clay holds firm, sandy soil doesn't form a ball. They also observe vegetation types growing naturally as indicators of soil type.",
                "source_document": "Karnataka ITK BOOK FINAL COPY 2020.pdf",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Procedural"
            },
        ]

        # ── Karnataka Traditional Seed Practices questions ───────────────────
        seed_practices_questions = [
            {
                "question": "What are the traditional methods used by Karnataka farmers for seed selection and preservation?",
                "expected_answer": "Karnataka farmers select seeds from healthy, vigorous, disease-free mother plants in the field itself. Seeds are sun-dried to remove moisture, treated with ash, dried neem leaves, or turmeric powder before storage. Seeds are stored in air-tight clay pots, gunny bags with ash lining, or bamboo containers. The selected seeds are kept separate from commercial seed lots and labeled with crop variety, season, and year. Community seed banks (beeja mitra) facilitate seed exchange between farmers.",
                "source_document": "Karnataka traditional seed practices knowledge.pdf",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Procedural"
            },
            {
                "question": "How is traditional seed priming performed in Karnataka farming communities?",
                "expected_answer": "Seed priming in Karnataka involves soaking seeds in water, diluted cow urine, or Bijamrita for 6-12 hours to improve germination rates. Seeds are then spread on mats under shade to drain and semi-dry before sowing. Some communities use ash water (soaking seeds in water mixed with wood ash) as an alkaline priming treatment. Seeds are sometimes wrapped in banana leaves or tender coconut leaves overnight before sowing to induce uniform sprouting.",
                "source_document": "Karnataka traditional seed practices knowledge.pdf",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Procedural"
            },
        ]

        # ── Indigenous Technical Knowledge general questions ─────────────────
        itk_general_questions = [
            {
                "question": "What is the definition and scope of Indigenous Technical Knowledge (ITK) in the context of Indian agriculture?",
                "expected_answer": "Indigenous Technical Knowledge (ITK) refers to the cumulative body of knowledge, practices, and beliefs about the relationship of living beings including humans with one another and with their environment, held by local communities, evolved by adaptive processes and handed down through generations by cultural transmission. In Indian agriculture, ITK encompasses traditional crop varieties, pest management, soil conservation, water management, livestock care (ethno-veterinary), food preservation, and medicinal plant use, developed and validated through centuries of community experience.",
                "source_document": "Indigenous Technical Knowledge.pdf",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Conceptual"
            },
            {
                "question": "Describe the ethno-veterinary practices documented as Indigenous Technical Knowledge for livestock disease management.",
                "expected_answer": "Ethno-veterinary ITK includes use of herbal preparations for common livestock ailments: neem bark decoction for fever, turmeric paste for wounds and infections, garlic and ajwain for digestive disorders, castor oil as a purgative, ginger and salt solution for bloat, and mustard oil massage for joint pain. Farmers use plant-based fly repellents, neem oil for external parasites, and smoke fumigation for ectoparasites. Traditional birth attendants (known as 'pashu vaidya') use specific root and bark preparations for difficult parturition and post-natal care.",
                "source_document": "Indigenous Technical Knowledge.pdf",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Domain-specific"
            },
            {
                "question": "How does ITK contribute to biodiversity conservation and why is its documentation important?",
                "expected_answer": "ITK contributes to biodiversity conservation by preserving traditional crop varieties (landraces) adapted to local conditions, maintaining diverse agroforestry systems, conserving medicinal plants through community use, and sustaining wild gene pools through traditional land management. Documentation is critical because: (1) rapid modernization erodes oral traditions within a generation, (2) ITK often holds solutions to emerging climate challenges, (3) biopiracy risks arise when undocumented TK is patented without community consent, and (4) integration of ITK with modern research can accelerate sustainable agricultural innovation.",
                "source_document": "Indigenous Technical Knowledge.pdf",
                "domain": "Traditional Knowledge / Sustainable Agriculture",
                "type": "Cultural"
            },
        ]

        # Return all questions combined
        all_questions = (
            manual_questions
            + icar_questions
            + karnataka_itk_questions
            + seed_practices_questions
            + itk_general_questions
        )
        return all_questions

    def generate_questions_with_gemini(self, chunks, doc_name, domain, num_questions=3, api_key=None):
        """Use Gemini to generate high-quality benchmark questions from document chunks."""
        key = api_key or config.get_gemini_api_key()
        if not key or not genai:
            print("Gemini API key not found or SDK not imported. Cannot generate questions using LLM.")
            return []

        genai.configure(api_key=key)
        
        # Select representative chunks randomly or by coverage
        sample_chunks = random.sample(chunks, min(len(chunks), num_questions * 2))
        combined_text = "\n\n--- CHUNK ---\n\n".join([c["text"] for c in sample_chunks])
        
        prompt = f"""You are a Responsible AI evaluator specializing in detecting knowledge gaps in LLMs.
Based on the following text extracts from a document named '{doc_name}' in the domain '{domain}', generate {num_questions} diverse evaluation questions.

TEXT EXTRACTS:
{combined_text}

For each question, formulate:
1. The question (targeting specific details, processes, or regional/cultural knowledge from the text).
2. The expected correct answer (comprehensive and exact, based on the text).
3. The type of question. Choose from: "Factual", "Procedural", "Contextual", "Domain-specific", "Cultural".

OUTPUT FORMAT:
Provide the output in strict JSON format as a list of objects. Do not wrap the JSON in ```json markdown formatting.
Each object must have exactly these keys:
"question": string,
"expected_answer": string,
"source_document": string (value must be "{doc_name}"),
"domain": string (value must be "{domain}"),
"type": string (one of the five types listed above)

Example JSON Output:
[
  {{
    "question": "Question text here?",
    "expected_answer": "Expected answer here.",
    "source_document": "{doc_name}",
    "domain": "{domain}",
    "type": "Factual"
  }}
]
"""
        try:
            model = genai.GenerativeModel(config.DEFAULT_GEMINI_MODEL)
            # Try to force JSON output
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"} if hasattr(genai, "GenerationConfig") else None
            )
            
            cleaned_text = response.text.strip()
            # Clean markdown code blocks if the model ignored instructions
            if cleaned_text.startswith("```"):
                cleaned_text = re.sub(r"^```json\s*", "", cleaned_text)
                cleaned_text = re.sub(r"^```\s*", "", cleaned_text)
                cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
            
            generated_questions = json.loads(cleaned_text)
            return generated_questions
        except Exception as e:
            print(f"Error generating questions with Gemini: {e}")
            return []

    def generate_benchmark(self, doc_id, num_questions=5, api_key=None):
        """
        Generate benchmark questions for a specific ingested document.
        Uses Gemini if possible, falls back to pre-defined questions for the sample document.
        """
        # Load document metadata to find chunks
        index_file = self.data_dir / "document_index.json"
        if not index_file.exists():
            return []
            
        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
            
        if doc_id not in index_data:
            print(f"Document ID {doc_id} not found in index.")
            return []
            
        doc_meta = index_data[doc_id]
        filename = doc_meta["filename"]
        domain = doc_meta["domain"]
        
        # Check if it is the sample document and we have no API key
        has_key = api_key or config.get_gemini_api_key()
        if filename == "traditional_agriculture_manual.txt" and not has_key:
            print("Using fallback questions for sample agriculture manual.")
            return self.get_fallback_questions(filename)[:num_questions]
            
        # Load full chunks
        chunks_file = self.data_dir / doc_meta["chunks_file"]
        if not chunks_file.exists():
            return []
            
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        # Try to generate via Gemini
        questions = self.generate_questions_with_gemini(
            chunks, filename, domain, num_questions, api_key
        )
        
        # If Gemini generation failed, and it's the sample document, return fallbacks
        if not questions and filename == "traditional_agriculture_manual.txt":
            print("Gemini generation failed or not configured. Falling back to pre-defined questions.")
            return self.get_fallback_questions(filename)[:num_questions]
            
        return questions

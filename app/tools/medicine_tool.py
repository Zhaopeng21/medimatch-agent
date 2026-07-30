import urllib.parse

from langchain_core.tools import tool

from app.config.settings import llm
from app.models.schemas import MedicinePrescription
from app.rag.vector_store import vector_db


@tool
def get_medicine_link(case_summary: str) -> str:
    """Recommend medicine for minor symptoms."""
    try:
        retrieved_context = ""
        if vector_db:
            docs = vector_db.similarity_search(case_summary, k=2)
            retrieved_context = "\n".join([doc.page_content for doc in docs])

        rag_prompt = (
            f"CONTEXT FROM INDIAN MEDICAL DATABASE:\n{retrieved_context}\n\n"
            f"STRUCTURED PATIENT SUMMARY:\n{case_summary}\n\n"
            f"YOUR TASK:\n"
            f"You are an expert clinical pharmacist cross-licensed in both India and New Zealand.\n"
            f"Use the structured patient summary as the patient presentation. Use its symptom, duration, and severity for your recommendation. Ignore conversational wording and do not extract symptoms from a user message.\n"
            f"Analyze the structured summary using the provided database context, extract the clinical solution, and safely map it to the New Zealand OTC market.\n\n"
            f"AGI TWO-STAGE REASONING PIPELINE:\n"
            f"1. CLINICAL EXTRACTION (Based on Database): Identify the standard medical treatment, core ACTIVE INGREDIENT (generic chemical name).\n"
            f"2. NZ LOCALIZATION MAPPING: Map that active ingredient to a mainstream commercial brand currently sitting on shelves in New Zealand pharmacies (like Chemist Warehouse NZ).\n\n"
            f"CRITICAL CROSS-BORDER & SAFETY MAPPING RULES:\n"
            f"- RULE 1 (ABSOLUTE): ROUTE OF ADMINISTRATION MATCHING. \n"
            f"  * Eye symptoms (dry/red eyes) REQUIRE Ophthalmic eye drops (e.g., Systane, Refresh, Optrex), NEVER topical skin creams.\n"
            f"  * Mouth ulcers (inside the mouth) REQUIRE Oral Mucosal Gels (e.g., Bonjela, SM33), NEVER topical skin pain gels (like Voltaren) on the outside of the face.\n"
            f"  * Chest heartburn REQUIRES oral liquid/tablets (e.g., Gaviscon).\n"
            f"- RULE 2: If the database suggests a foreign brand, DO NOT present it. Translate it to its active generic ingredient first, then find the iconic NZ brand equivalent.\n"
            f"- RULE 3: The 'search_keyword' field must contain ONLY the mapped NZ commercial brand name (1-2 words), nothing else.\n\n"
            f"First, use the 'clinical_reasoning_scratchpad' to lock in the anatomy and route. Then, output the matching brand."
        )

        structured_llm = llm.with_structured_output(MedicinePrescription)
        result = structured_llm.invoke(rag_prompt)

        clean_brand = result.search_keyword.strip().replace("*", "").replace(".", "")
        link = f"https://www.chemistwarehouse.co.nz/search?searchtext={urllib.parse.quote(clean_brand)}"

        return (
            f"💝 **Minor Case Advice: {clean_brand}**\n\n"
            f"{result.clinical_advice}\n\n"
            f"🛒 **Buy Online**: [{clean_brand} on Chemist Warehouse]({link})"
        )
    except Exception as e:
        return f"Please consult a local pharmacist in Auckland. (Error: {str(e)})"

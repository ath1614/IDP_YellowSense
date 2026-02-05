import asyncio
import aiohttp
import json

async def test_complete_pipeline():
    async with aiohttp.ClientSession() as session:
        print("=== GOVERNMENT IDP SYSTEM TEST ===")
        
        # Step 1: OCR Processing
        print("\n1. Processing PDF with OCR...")
        with open('one_page_test.pdf', 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename='test.pdf', content_type='application/pdf')
            
            async with session.post("http://34.180.7.209:8000/ocr", data=data) as resp:
                ocr_result = await resp.json()
                full_text = " ".join([page["text"] for page in ocr_result["pages"]])
                print(f"✓ OCR Complete: {len(full_text)} characters extracted")
        
        # Step 2: Document Classification (using SSH tunnel on port 8002)
        print("\n2. Classifying document...")
        async with session.post(
            "http://localhost:8002/classify",
            json={"text": full_text}
        ) as resp:
            classification = await resp.json()
            doc_type = classification["document_type"]
            print(f"✓ Classification: {doc_type}")
        
        # Step 3: Extract structured data
        print(f"\n3. Extracting {doc_type} data...")
        endpoint = "apar" if doc_type == "APAR" else "disciplinary"
        async with session.post(
            f"http://localhost:8002/{endpoint}",
            json={"text": full_text}
        ) as resp:
            structured_data = await resp.json()
            print("✓ Structured Data:")
            print(json.dumps(structured_data, indent=2))
        
        print("\n🎉 GOVERNMENT IDP SYSTEM WORKING!")

asyncio.run(test_complete_pipeline())

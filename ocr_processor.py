"""
OCR Processor using GPT-4 Vision for scanned documents
"""
import os
import base64
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
from PIL import Image

import io
from dotenv import load_dotenv

load_dotenv()

class OCRProcessor:
    """Process images and scanned documents using GPT-4 Vision"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"  # GPT-4 with vision
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def process_image(self, image_path: str, language: str = "Arabic and English") -> Dict:
        """
        Extract text from image using GPT-4 Vision
        
        Args:
            image_path: Path to image file
            language: Expected language in image
            
        Returns:
            Dict with extracted text and metadata
        """
        try:
            # Encode image
            base64_image = self.encode_image(image_path)
            
            # Create prompt
            prompt = f"""Extract ALL text from this image in {language}. 

Rules:
1. Preserve the original layout and structure
2. If there are tables, format them clearly
3. If there are multiple columns, separate them with |
4. Include ALL text, even small details
5. If text is in Arabic, keep it in Arabic
6. If text is unclear, note it with [unclear]

Return ONLY the extracted text, no explanations."""
            
            # Call GPT-4 Vision
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            extracted_text = response.choices[0].message.content
            
            return {
                'text': extracted_text,
                'status': 'success',
                'method': 'gpt4_vision',
                'model': self.model
            }
            
        except Exception as e:
            return {
                'text': '',
                'status': 'error',
                'error': str(e),
                'method': 'gpt4_vision'
            }
    
    def process_pdf_page_as_image(self, pdf_path: str, page_num: int = 0) -> Dict:
        """
        Convert PDF page to image and extract text
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number to process (0-indexed)
            
        Returns:
            Dict with extracted text
        """
        try:
            import fitz  # PyMuPDF
            
            # Open PDF
            doc = fitz.open(pdf_path)
            
            if page_num >= len(doc):
                return {
                    'text': '',
                    'status': 'error',
                    'error': f'Page {page_num} does not exist'
                }
            
            # Get page
            page = doc[page_num]
            
            # Convert to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
            img_data = pix.tobytes("png")
            
            # Save temporarily
            temp_path = f"temp_page_{page_num}.png"
            with open(temp_path, "wb") as f:
                f.write(img_data)
            
            # Process with OCR
            result = self.process_image(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            doc.close()
            
            return result
            
        except Exception as e:
            return {
                'text': '',
                'status': 'error',
                'error': str(e)
            }
    
    def process_scanned_pdf(self, pdf_path: str, max_pages: int = 3) -> Dict:
        """
        Process entire scanned PDF
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to process
            
        Returns:
            Dict with all extracted text
        """
        try:
            import fitz
            
            doc = fitz.open(pdf_path)
            num_pages = min(len(doc), max_pages)
            
            all_text = []
            pages_data = []
            
            for page_num in range(num_pages):
                result = self.process_pdf_page_as_image(pdf_path, page_num)
                
                if result['status'] == 'success':
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': result['text'],
                        'char_count': len(result['text'])
                    })
                    all_text.append(f"[Page {page_num + 1}]\n{result['text']}")
            
            doc.close()
            
            return {
                'text': '\n\n'.join(all_text),
                'pages': pages_data,
                'status': 'success',
                'method': 'gpt4_vision_ocr',
                'num_pages': num_pages
            }
            
        except Exception as e:
            return {
                'text': '',
                'pages': [],
                'status': 'error',
                'error': str(e)
            }
    
    def detect_if_scanned(self, pdf_path: str) -> bool:
        """
        Detect if PDF is scanned (image-based) or text-based
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if scanned, False if text-based
        """
        try:
            import fitz
            
            doc = fitz.open(pdf_path)
            
            # Check first page
            page = doc[0]
            text = page.get_text()
            
            # If very little text, likely scanned
            is_scanned = len(text.strip()) < 500  #50
            
            doc.close()
            
            return is_scanned
            
        except Exception as e:
            print(f"Error detecting PDF type: {e}")
            return False


# Test function
if __name__ == "__main__":
    ocr = OCRProcessor()
    
    # Test with sample image
    test_image = "test_image.png"
    if os.path.exists(test_image):
        result = ocr.process_image(test_image)
        print(f"Status: {result['status']}")
        print(f"Extracted text length: {len(result.get('text', ''))}")
        print(f"Text preview: {result.get('text', '')[:200]}")
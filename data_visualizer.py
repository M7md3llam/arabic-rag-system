"""
Data visualization module for generating tables and charts
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from typing import List, Dict, Optional
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class DataVisualizer:
    """Generate tables and charts from data"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract_structured_data(self, query: str, context: str) -> Optional[Dict]:
        """
        Use GPT to extract structured data from context
        
        Args:
            query: User's request
            context: Retrieved document content
            
        Returns:
            Dict with structured data or None
        """
        try:
            prompt = f"""Based on this query: "{query}"

Extract structured data from the following context and return it as JSON.

Context:
{context}

If the query asks for a comparison, table, or chart, extract the relevant data.
Return ONLY valid JSON in this format:
{{
    "type": "table" or "chart",
    "title": "descriptive title",
    "data": [
        {{"label": "item1", "value": 100, "category": "optional"}},
        {{"label": "item2", "value": 200, "category": "optional"}}
    ],
    "chart_type": "bar" or "line" or "pie" (only if type is chart)
}}

If no structured data can be extracted, return: {{"type": "none"}}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract structured data and return valid JSON only. No explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean up response (remove markdown code blocks if present)
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            
            if result.get("type") == "none":
                return None
            
            return result
            
        except Exception as e:
            print(f"Error extracting structured data: {e}")
            return None
    
    def create_table(self, data: List[Dict], title: str = "Data Table") -> pd.DataFrame:
        """
        Create a pandas DataFrame from structured data
        
        Args:
            data: List of dicts with data
            title: Table title
            
        Returns:
            pandas DataFrame
        """
        try:
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            print(f"Error creating table: {e}")
            return pd.DataFrame()
    
    def create_chart(self, 
                     data: List[Dict], 
                     chart_type: str = "bar",
                     title: str = "Chart",
                     xlabel: str = "",
                     ylabel: str = "Value") -> Optional[str]:
        """
        Create a chart and return as base64 encoded image
        
        Args:
            data: List of dicts with 'label' and 'value'
            chart_type: 'bar', 'line', or 'pie'
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Base64 encoded PNG image string
        """
        try:
            # Validate data
            if not data or len(data) == 0:
                print("No data provided for chart")
                return None
            
            # Extract labels and values with better error handling
            labels = []
            values = []
            
            for i, item in enumerate(data):
                label = item.get('label', item.get('name', f'Item {i+1}'))
                value = item.get('value', item.get('count', 0))
                
                # Try to convert value to float
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    print(f"Invalid value for {label}: {value}")
                    continue
                
                labels.append(str(label))
                values.append(value)
            
            if not labels or not values:
                print("No valid data extracted for chart")
                return None
            
            # Create figure with better styling
            fig, ax = plt.subplots(figsize=(12, 7))
            fig.patch.set_facecolor('white')
            
            # Color palette
            colors = ['#4A90E2', '#50C878', '#FF6B6B', '#FFA500', '#9B59B6', 
                     '#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#1ABC9C']
            
            if chart_type == "bar":
                bars = ax.bar(labels, values, color=colors[:len(labels)], 
                             edgecolor='white', linewidth=1.5, alpha=0.8)
                ax.set_xlabel(xlabel if xlabel else "Categories", fontsize=11, fontweight='bold')
                ax.set_ylabel(ylabel if ylabel else "Values", fontsize=11, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=9)
                
            elif chart_type == "line":
                ax.plot(labels, values, marker='o', linewidth=3, markersize=10,
                       color=colors[0], markerfacecolor=colors[1], 
                       markeredgecolor='white', markeredgewidth=2)
                ax.set_xlabel(xlabel if xlabel else "Categories", fontsize=11, fontweight='bold')
                ax.set_ylabel(ylabel if ylabel else "Values", fontsize=11, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add value labels
                for i, (x, y) in enumerate(zip(labels, values)):
                    ax.text(i, y, f'{y:.1f}', ha='center', va='bottom', fontsize=9)
                
            elif chart_type == "pie":
                wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                                   startangle=90, colors=colors[:len(labels)],
                                                   wedgeprops={'edgecolor': 'white', 'linewidth': 2})
                ax.axis('equal')
                
                # Make percentage text bold
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Remove top and right spines for bar/line charts
            if chart_type in ["bar", "line"]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return img_base64
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_comparison_table(self, 
                                 query: str,
                                 documents: List[str],
                                 metadatas: List[Dict]) -> Optional[pd.DataFrame]:
        """
        Generate a comparison table from multiple documents
        
        Args:
            query: Comparison query
            documents: List of document chunks
            metadatas: Metadata for each chunk
            
        Returns:
            pandas DataFrame with comparison
        """
        try:
            # Combine documents with their sources
            context = "\n\n".join([
                f"[Source: {meta.get('document_name', 'Unknown')}]\n{doc}"
                for doc, meta in zip(documents, metadatas)
            ])
            
            # Extract structured data
            structured_data = self.extract_structured_data(query, context)
            
            if structured_data and structured_data.get("type") == "table":
                df = self.create_table(
                    structured_data.get("data", []),
                    structured_data.get("title", "Comparison Table")
                )
                return df
            
            return None
            
        except Exception as e:
            print(f"Error generating comparison table: {e}")
            return None


# Test function
if __name__ == "__main__":
    visualizer = DataVisualizer()
    
    # Test data
    test_data = [
        {"label": "Product A", "value": 100},
        {"label": "Product B", "value": 150},
        {"label": "Product C", "value": 80}
    ]
    
    # Test table
    df = visualizer.create_table(test_data, "Test Table")
    print("Table created:")
    print(df)
    
    # Test chart
    chart_img = visualizer.create_chart(test_data, chart_type="bar", title="Test Chart")
    if chart_img:
        print(f"Chart created (base64 length: {len(chart_img)})")
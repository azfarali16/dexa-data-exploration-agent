import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import google.generativeai as genai
import json
import uuid
from utils.visualizer import Visualizer

class InsightAgent:
    """Autonomous agent that explores data and generates insights"""
    
    def __init__(self, dataframe: pd.DataFrame, api_key: str):
        """
        Initialize the insight agent with a dataframe and API key.
        
        Args:
            dataframe: Pandas DataFrame to analyze
            api_key: Gemini API key
        """
        self.df = dataframe
        self.api_key = api_key
        self.visualizer = Visualizer(dataframe)
        
        # Configure Gemini

        genai.configure(api_key= api_key)
        self.model = genai.GenerativeModel(
                            'gemini-2.0-flash',
                            system_instruction="You are Dexa, an AI-powered data exploration system that profiles datasets, detects patterns and anomalies, suggests visualizations and next steps, and delivers a visual and text-based action plan. You will answer user queries and propose follow-up questions to help uncover deeper, more insightful findings.",
                            generation_config=genai.GenerationConfig(
                                response_mime_type="application/json",
                                temperature = 0.3
                            ),
                        )


    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate automatic insights from the data.
        
        Returns:
            Dict: Insights, findings, and suggested questions
        """
        try:
            # Get dataframe info
            schema = self._get_schema()
            profile = self._get_data_profile()
            
            # Generate insights using LLM
            insights_prompt = self._construct_insights_prompt(schema, profile)
            response = self.model.generate_content(insights_prompt)
            insights_text = response.text
            
            # Parse insights
            parsed_insights = self._parse_insights(insights_text)
            
            # Generate visualizations for insights
            self._add_visualizations_to_insights(parsed_insights)
            
            return parsed_insights
            
        except Exception as e:
            # Return error with minimal structure
            return {
                'summary': f"Error generating insights: {str(e)}",
                'findings': [],
                'suggested_questions': [
                    "What columns are in this dataset?",
                    "Can you show me basic statistics for the data?",
                    "What are the relationships between the main variables?"
                ]
            }
    
    def _get_schema(self) -> str:
        """Get the dataframe schema as a string"""
        types = self.df.dtypes.to_dict()
        
        schema = "DataFrame Schema:\n"
        for column, dtype in types.items():
            schema += f"- {column} ({dtype})\n"
        
        schema += f"\nRows: {len(self.df)}\n"
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        date_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        schema += f"Numeric columns: {', '.join(num_cols)}\n"
        schema += f"Categorical columns: {', '.join(cat_cols)}\n"
        schema += f"Date columns: {', '.join(date_cols)}\n"
        
        return schema
    
    def _get_data_profile(self) -> str:
        """Generate a profile of the data"""
        profile = "Data Profile:\n"
        
        # Basic statistics for numeric columns
        num_cols = self.df.select_dtypes(include=[np.number])
        if not num_cols.empty:
            profile += "Numeric Columns Stats:\n"
            profile += num_cols.describe().to_string() + "\n\n"
        
        # Top values for categorical columns
        cat_cols = self.df.select_dtypes(include=['object'])
        if not cat_cols.empty:
            profile += "Categorical Columns (top values):\n"
            for col in cat_cols.columns[:5]:  # Limit to first 5 categorical columns
                value_counts = self.df[col].value_counts().nlargest(5)
                profile += f"{col}:\n{value_counts.to_string()}\n\n"
        
        # Check for missing values
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            profile += "Missing Values:\n"
            profile += missing.to_string() + "\n\n"
        
        # Correlations between numeric columns
        if len(num_cols.columns) > 1:
            profile += "Correlations:\n"
            profile += self.df[num_cols.columns].corr().round(2).to_string() + "\n\n"
        
        return profile
    
    def _construct_insights_prompt(self, schema: str, profile: str) -> str:
        """Construct a prompt for the insights generation"""
        prompt = f"""You are an AI data scientist assistant that automatically analyzes datasets to find interesting patterns and insights.
Analyze the following dataset information and generate key insights.

{schema}

{profile}

Your task is to:
1. Identify the most important patterns, trends, and relationships in this data
2. Detect any anomalies or outliers
3. Determine what visualizations would be most relevant
4. Suggest actionable insights based on the data
5. Generate follow-up questions users might want to ask

Output in JSON format:
```json
{{
  "summary": "One paragraph summary of the key insights",
  "findings": [
    {{
      "id": "unique_id",
      "title": "Brief title of finding",
      "description": "2-3 sentences explaining the insight",
      "importance": "high|medium|low",
      "has_visualization": true/false,
      "visualization_type": "distribution|comparison|trend|scatter",
      "visualization_columns": ["column1", "column2"],
      "recommended_action": "A specific action or follow-up question"
    }},
    // Add 3-5 more findings
  ],
  "suggested_questions": [
    "Question 1 about the data?",
    "Question 2 about the data?",
    "Question 3 about the data?",
    "Question 4 about the data?",
    "Question 5 about the data?"
  ]
}}
```

Make sure the insights are:
- Data-driven and specific to this dataset
- Actionable and useful for decision-making
- Diverse, covering different aspects of the data
- Expressed in simple, non-technical language"""

        return prompt
    
    def _parse_insights(self, insights_text: str) -> Dict[str, Any]:
        """Parse the insights response from the model"""
        # Extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'```json\n(.*?)```', insights_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no code block, try to use the entire text
                json_str = insights_text
            
            # Parse JSON
            insights = json.loads(json_str)
            
            # Ensure proper structure
            if 'summary' not in insights:
                insights['summary'] = "Analysis complete. Here are the key findings from your data."
                
            if 'findings' not in insights or not insights['findings']:
                insights['findings'] = [{
                    'id': str(uuid.uuid4()),
                    'title': 'Basic Data Overview',
                    'description': f"Your dataset has {len(self.df)} rows and {len(self.df.columns)} columns.",
                    'importance': 'medium',
                    'has_visualization': False,
                    'recommended_action': 'Explore basic statistics'
                }]
                
            if 'suggested_questions' not in insights or not insights['suggested_questions']:
                insights['suggested_questions'] = [
                    "What columns are in this dataset?",
                    "Can you show me basic statistics for the data?",
                    "What are the relationships between the main variables?"
                ]
            
            # Assign IDs to findings if missing
            for finding in insights['findings']:
                if 'id' not in finding:
                    finding['id'] = str(uuid.uuid4())
            
            return insights
            
        except Exception as e:
            # Return minimal structure on error
            return {
                'summary': "Analysis complete. Here are some observations about your data.",
                'findings': [{
                    'id': str(uuid.uuid4()),
                    'title': 'Basic Data Overview',
                    'description': f"Your dataset has {len(self.df)} rows and {len(self.df.columns)} columns.",
                    'importance': 'medium',
                    'has_visualization': False,
                    'recommended_action': 'Explore basic statistics'
                }],
                'suggested_questions': [
                    "What columns are in this dataset?",
                    "Can you show me basic statistics for the data?",
                    "What are the relationships between the main variables?"
                ]
            }
    
    def _add_visualizations_to_insights(self, insights: Dict[str, Any]) -> None:
        """Add visualizations to insights based on findings"""
        for finding in insights['findings']:
            # Check if visualization is needed
            if finding.get('has_visualization', False) and 'visualization_columns' in finding:
                columns = finding['visualization_columns']
                viz_type = finding.get('visualization_type', 'distribution')
                
                # Add visualization only if we have valid columns
                if columns and all(col in self.df.columns for col in columns):
                    # Create chart
                    chart = self.visualizer.create_chart_for_query(
                        viz_type,
                        columns
                    )
                    finding['visualization'] = chart.get('config', {})
                else:
                    finding['has_visualization'] = False
    
    def process_conversation(self, user_message: str) -> Dict[str, Any]:
        """
        Process a follow-up question or request from the user.
        
        Args:
            user_message: User's question or request
            
        Returns:
            Dict: Response with text and optional visualization
        """
        try:
            # Get dataframe schema
            schema = self._get_schema()
            
            # Construct prompt for conversation
            prompt = f"""You are an AI data assistant analyzing the following dataset:

{schema}

The user asks: "{user_message}"

Respond to the user's query with:
1. A clear, helpful answer based on the data
2. If appropriate, specify a visualization to show using this format:

```visualization
{{
  "type": "<visualization_type>",  // distribution, comparison, trend, scatter
  "columns": ["column1", "column2"],  // 1-2 columns to visualize
  "filters": {{  // Optional filters
    "column_name": {{
      "min": value,
      "max": value,
      "equals": value,
      "in": [value1, value2]
    }}
  }}
}}
```

Keep your response conversational but concise. Focus on directly answering the query."""

            # Generate response
            response = self.model.generate_content(prompt)
            result = response.text
            
            # Parse the response
            visualization_config = self._extract_visualization_config(result)
            text_response = self._extract_text_response(result)
            
            # Prepare final response
            final_response = {
                'response': text_response,
                'has_visualization': bool(visualization_config)
            }
            
            # Add visualization if available
            if visualization_config:
                chart = self.visualizer.create_chart_for_query(
                    visualization_config.get('type', 'distribution'),
                    visualization_config.get('columns', []),
                    visualization_config.get('filters')
                )
                final_response['visualization'] = chart.get('config', {})
            
            return final_response
            
        except Exception as e:
            return {
                'response': f"I'm sorry, I couldn't process that request. Error: {str(e)}",
                'has_visualization': False
            }
    
    def _extract_visualization_config(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract visualization config from model response"""
        # Look for visualization JSON block
        match = re.search(r'```visualization\n(.*?)```', response, re.DOTALL)
        
        if not match:
            return None
            
        try:
            # Parse the JSON
            json_str = match.group(1).strip()
            config = json.loads(json_str)
            return config
        except:
            return None
    
    def _extract_text_response(self, response: str) -> str:
        """Extract the text response without the visualization block"""
        # Remove visualization block
        clean_response = re.sub(r'```visualization\n.*?```', '', response, flags=re.DOTALL)
        
        # Remove any other code blocks
        clean_response = re.sub(r'```.*?```', '', clean_response, flags=re.DOTALL)
        
        # Clean up and format
        return clean_response.strip() 
import json
from utils.logger import log_error,setup_logger
import re

logger = setup_logger(__name__)
def _safe_json_extract(agent_output) -> dict:
    """Safely extract JSON from agent output - Complete Single Function"""
    try:
        # Get the content
        content = agent_output.content if hasattr(agent_output, "content") else agent_output
        
        # Case 1: If content is already a dict, check if it contains FilterSchema
        if isinstance(content, dict):
            if 'text' in content:
                # This is your problematic case - extract the FilterSchema
                filter_schema = content['text']
                
                # If it's a FilterSchema object, convert to dict
                if hasattr(filter_schema, '__dict__'):
                    return filter_schema.__dict__
                elif isinstance(filter_schema, dict):
                    return filter_schema
                else:
                    # Parse FilterSchema string representation
                    text = str(filter_schema)
                    if 'FilterSchema(' in text:
                        # Extract content between parentheses
                        start = text.find('FilterSchema(') + 13
                        end = text.rfind(')')
                        if start < end:
                            params_str = text[start:end]
                            
                            # Parse parameters
                            filters = {}
                            # Split by comma but handle nested structures
                            params = []
                            current_param = ""
                            bracket_count = 0
                            quote_count = 0
                            
                            for char in params_str:
                                if char == '[':
                                    bracket_count += 1
                                elif char == ']':
                                    bracket_count -= 1
                                elif char == "'":
                                    quote_count = 1 - quote_count
                                elif char == ',' and bracket_count == 0 and quote_count == 0:
                                    params.append(current_param.strip())
                                    current_param = ""
                                    continue
                                current_param += char
                            
                            if current_param.strip():
                                params.append(current_param.strip())
                            
                            # Parse each parameter
                            for param in params:
                                if '=' in param:
                                    key, value = param.split('=', 1)
                                    key = key.strip()
                                    value = value.strip()
                                    
                                    # Parse the value
                                    if value == 'None':
                                        filters[key] = None
                                    elif value == '[]':
                                        filters[key] = []
                                    elif value.startswith('[') and value.endswith(']'):
                                        # Parse list
                                        try:
                                            items_str = value[1:-1]  # Remove brackets
                                            if not items_str.strip():
                                                filters[key] = []
                                            else:
                                                items = []
                                                for item in items_str.split(','):
                                                    item = item.strip()
                                                    if item.startswith("'") and item.endswith("'"):
                                                        items.append(item[1:-1])
                                                    elif item.startswith('"') and item.endswith('"'):
                                                        items.append(item[1:-1])
                                                    else:
                                                        items.append(item)
                                                filters[key] = items
                                        except:
                                            filters[key] = []
                                    elif value.startswith("'") and value.endswith("'"):
                                        filters[key] = value[1:-1]
                                    elif value.startswith('"') and value.endswith('"'):
                                        filters[key] = value[1:-1]
                                    elif value.isdigit():
                                        filters[key] = int(value)
                                    elif value.replace('.', '').replace('-', '').isdigit():
                                        filters[key] = float(value)
                                    else:
                                        filters[key] = value
                            
                            return filters
                    return {}
            else:
                # Regular dict, return as is
                return content
        
        # Case 2: If content is a string, try JSON parsing
        elif isinstance(content, str):
            # Remove markdown formatting
            clean_content = re.sub(r"```(?:json)?|```", "", content).strip()
            
            # Try direct JSON parsing first
            try:
                return json.loads(clean_content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to parse as FilterSchema string
                if 'FilterSchema(' in clean_content:
                    # Extract content between parentheses
                    start = clean_content.find('FilterSchema(') + 13
                    end = clean_content.rfind(')')
                    if start < end:
                        params_str = clean_content[start:end]
                        
                        # Parse parameters (same logic as above)
                        filters = {}
                        params = []
                        current_param = ""
                        bracket_count = 0
                        quote_count = 0
                        
                        for char in params_str:
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                            elif char == "'":
                                quote_count = 1 - quote_count
                            elif char == ',' and bracket_count == 0 and quote_count == 0:
                                params.append(current_param.strip())
                                current_param = ""
                                continue
                            current_param += char
                        
                        if current_param.strip():
                            params.append(current_param.strip())
                        
                        # Parse each parameter
                        for param in params:
                            if '=' in param:
                                key, value = param.split('=', 1)
                                key = key.strip()
                                value = value.strip()
                                
                                # Parse the value
                                if value == 'None':
                                    filters[key] = None
                                elif value == '[]':
                                    filters[key] = []
                                elif value.startswith('[') and value.endswith(']'):
                                    # Parse list
                                    try:
                                        items_str = value[1:-1]  # Remove brackets
                                        if not items_str.strip():
                                            filters[key] = []
                                        else:
                                            items = []
                                            for item in items_str.split(','):
                                                item = item.strip()
                                                if item.startswith("'") and item.endswith("'"):
                                                    items.append(item[1:-1])
                                                elif item.startswith('"') and item.endswith('"'):
                                                    items.append(item[1:-1])
                                                else:
                                                    items.append(item)
                                            filters[key] = items
                                    except:
                                        filters[key] = []
                                elif value.startswith("'") and value.endswith("'"):
                                    filters[key] = value[1:-1]
                                elif value.startswith('"') and value.endswith('"'):
                                    filters[key] = value[1:-1]
                                elif value.isdigit():
                                    filters[key] = int(value)
                                elif value.replace('.', '').replace('-', '').isdigit():
                                    filters[key] = float(value)
                                else:
                                    filters[key] = value
                        
                        return filters
                # If not FilterSchema format, return empty dict
                return {}
        
        # Case 3: If it's a FilterSchema object directly
        elif hasattr(content, '__dict__'):
            return content.__dict__
        
        # Case 4: Fallback - return empty dict
        else:
            logger.warning(f"⚠️ Unknown content type: {type(content)}")
            return {}
            
    except Exception as e:
        logger.error(f"❌ JSON extraction error: {str(e)}")
        logger.error(f"Raw content: {content}")
        logger.error(f"Content type: {type(content)}")
        return {}
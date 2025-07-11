# utils/code_executor.py
import pandas as pd
import traceback
from typing import Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CodeExecutor:
    """Safe execution of pandas code with comprehensive error handling"""
    
    def __init__(self):
        self.safe_globals = {
            'pd': pd,
            '__builtins__': {
                'len': len, 
                'str': str, 
                'int': int, 
                'float': float,
                'list': list,
                'dict': dict,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sorted': sorted,
                'max': max,
                'min': min,
                'sum': sum,
                'any': any,
                'all': all
            }
        }
    
    def execute_filter(self, df: pd.DataFrame, pandas_code: str) -> pd.DataFrame:
        """Execute pandas filter code safely"""
        try:
            logger.info(f"🔍 Executing pandas code: {pandas_code[:100]}...")
            
            # Validate code for basic safety
            if not self._is_code_safe(pandas_code):
                logger.warning("❌ Code validation failed - potentially unsafe operations detected")
                return pd.DataFrame()
            
            # Create execution environment
            local_vars = {"df": df.copy()}
            
            # Execute code
            exec(pandas_code, self.safe_globals, local_vars)
            
            # Find the result DataFrame
            result_df = self._extract_result_dataframe(local_vars)
            
            logger.info(f"✅ Filter executed successfully: {len(result_df)} results")
            return result_df
            
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in pandas code: {str(e)}")
            logger.error(f"Code: {pandas_code}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Pandas execution error: {str(e)}")
            logger.error(f"Code that failed: {pandas_code}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _is_code_safe(self, code: str) -> bool:
        """Basic safety check for pandas code"""
        # List of dangerous operations
        dangerous_operations = [
            'import', 'exec', 'eval', 'open', 'file', 'input', 'raw_input',
            'subprocess', 'os.', 'sys.', '__import__', 'compile', 'globals',
            'locals', 'vars', 'dir', 'hasattr', 'getattr', 'setattr', 'delattr'
        ]
        
        code_lower = code.lower()
        
        for operation in dangerous_operations:
            if operation in code_lower:
                logger.warning(f"❌ Dangerous operation detected: {operation}")
                return False
        
        return True
    
    def _extract_result_dataframe(self, local_vars: Dict[str, Any]) -> pd.DataFrame:
        """Extract the result DataFrame from execution variables"""
        # Look for common result variable names
        result_names = ['filtered_df', 'result_df', 'df_filtered', 'df_result', 'result']
        
        for name in result_names:
            if name in local_vars and isinstance(local_vars[name], pd.DataFrame):
                return local_vars[name]
        
        # If no specific result found, look for any DataFrame except the original
        for var_name, var_value in local_vars.items():
            if (isinstance(var_value, pd.DataFrame) and 
                var_name != 'df' and 
                not var_name.startswith('_')):
                return var_value
        
        # If nothing found, return the modified original DataFrame
        return local_vars.get('df', pd.DataFrame())
    
    def validate_and_fix_code(self, code: str) -> str:
        """Validate and attempt to fix common issues in pandas code"""
        try:
            # Remove any markdown formatting
            code = code.strip()
            if code.startswith('```python'):
                code = code[9:]
            if code.startswith('```'):
                code = code[3:]
            if code.endswith('```'):
                code = code[:-3]
            
            # Ensure the code assigns to a result variable
            if 'filtered_df' not in code and 'result_df' not in code:
                # If no explicit assignment, add one
                lines = code.split('\n')
                if lines and not any('=' in line for line in lines):
                    code = f"filtered_df = df.{code}"
            
            return code.strip()
            
        except Exception as e:
            logger.error(f"❌ Error validating code: {str(e)}")
            return code
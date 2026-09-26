"""Synthetic CLI fixture: reports paths only, never credential values."""
import json
import sys
from submission_paths import PACKAGE_ROOT, path
sys.path.insert(0,str(PACKAGE_ROOT/'analysis/bcr'))
import bcr_inputs
from code_strategy3.python_settings import WORKSPACE_DIR
print(json.dumps({'project':str(path('project')),'data':str(path('data')),
                  'models':str(path('models')),'bcr_clinical':str(bcr_inputs.CLINICAL_PATH),
                  'llm_workspace':str(WORKSPACE_DIR)}))

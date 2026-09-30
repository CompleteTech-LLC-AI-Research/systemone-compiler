"""No DSPy, GEPA, or vendor SDK imports occur at runtime package import."""
from .models import UseCase, Program, Decision, StateField
from .hierarchy import HierarchySource, HierarchyArtifact, lower_hierarchy, load_artifact
from .hierarchy_validation import (validate_hierarchy_artifact, validate_hierarchy_source,
                                   validate_hierarchy_compile_inputs)
from .hierarchy_runtime import HierarchyRuntime
from .hierarchy_data import (HierarchyExample, IntermediateAnnotation, HierarchySplitGuard,
                             HierarchyTeacherInputs, read_hierarchy_jsonl)
from .runtime import Runtime
from .compiler import Compiler, CompileOptions

__version__ = "0.1.0"
__all__ = ["UseCase", "Program", "Decision", "StateField", "Runtime", "Compiler", "CompileOptions",
           "HierarchySource", "HierarchyArtifact", "lower_hierarchy", "load_artifact",
           "validate_hierarchy_artifact", "validate_hierarchy_source", "validate_hierarchy_compile_inputs",
           "HierarchyRuntime", "HierarchyExample", "IntermediateAnnotation",
           "HierarchySplitGuard", "HierarchyTeacherInputs", "read_hierarchy_jsonl"]

# Documentation

This directory contains specifications and documentation for the RecompTracker Body Composition Analysis web application.

## Contents

### [v1-url-state-spec.md](v1-url-state-spec.md)

The V1 URL State Specification defines the baseline format for sharing application state via URLs. This specification was established when the application was first released publicly and serves as the foundation for backward compatibility.

**Key Features:**
- Complete URL state format definition
- Compact JSON schema with examples
- Encoding/decoding process documentation
- Validation rules and constraints
- Backward compatibility commitment

### [v1-url-state-schema.json](v1-url-state-schema.json)

Programmatic JSON Schema specification for the V1 URL state format. This machine-readable schema enables:
- Automated validation of V1 URL data
- Integration with development tools and testing
- Consistent validation across different components
- Documentation generation from schema

### [v1_schema_validator.py](v1_schema_validator.py)

Python utilities for validating V1 URL state data against the JSON Schema. Provides:
- `validate_v1_config()` - Validate compact JSON configs
- `validate_v1_url()` - Validate complete URLs with decoding
- `get_schema_errors()` - Get detailed validation errors
- Command-line interface for manual validation

This specification ensures that URLs shared by users will continue to work as the application evolves, providing a stable public interface for data sharing.

## Purpose

These specifications serve to:

1. **Document Public Interfaces**: Establish clear contracts for features exposed to users
2. **Enable Backward Compatibility**: Provide reference for maintaining compatibility with shared URLs
3. **Guide Development**: Ensure future changes consider impact on existing user data
4. **Support Testing**: Define requirements for comprehensive compatibility testing

## Maintenance

Specifications should be updated when:
- New URL format versions are introduced
- Breaking changes are made to existing formats
- Additional validation rules are implemented
- New features are added that affect URL state

All changes should maintain backward compatibility with previous versions unless absolutely necessary for security or critical functionality.
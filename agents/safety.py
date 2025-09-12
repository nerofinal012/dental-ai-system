# agents/safety.py
"""
Safety agent for PHI protection and compliance
"""

import re
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SafetyAgent:
    """Agent for PHI detection, redaction, and compliance"""
    
    def __init__(self):
        """Initialize safety agent"""
        self.name = "Safety"
        self.description = "PHI protection and compliance specialist"
        
        # PHI patterns
        self.phi_patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b(?:\+?1[-.]?)?\(?[2-9]\d{2}\)?[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'dob': r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b',
            'mrn': r'\b(?:MRN|mrn|Medical Record Number)[:\s#]*\d{6,}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        }
        
        # Access control matrix
        self.access_matrix = {
            'patient': {
                'allowed': ['own_records', 'appointments', 'billing'],
                'denied': ['other_patients', 'admin_functions', 'system_config']
            },
            'staff': {
                'allowed': ['patient_records', 'appointments', 'billing', 'reports'],
                'denied': ['system_config', 'audit_logs']
            },
            'admin': {
                'allowed': ['*'],  # Full access
                'denied': []
            }
        }
        
        self.audit_log = []
    
    def detect_phi(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PHI in text
        
        Args:
            text: Text to scan
            
        Returns:
            Dictionary of detected PHI types and matches
        """
        detected = {}
        
        for phi_type, pattern in self.phi_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected[phi_type] = matches
                logger.warning(f"PHI detected: {phi_type} ({len(matches)} instances)")
        
        return detected
    
    def redact_phi(self, text: str, replacement_format: str = "[{}_REDACTED]") -> str:
        """
        Redact PHI from text
        
        Args:
            text: Text to redact
            replacement_format: Format for replacement text
            
        Returns:
            Redacted text
        """
        redacted_text = text
        
        for phi_type, pattern in self.phi_patterns.items():
            replacement = replacement_format.format(phi_type.upper())
            redacted_text = re.sub(pattern, replacement, redacted_text, flags=re.IGNORECASE)
        
        return redacted_text
    
    def check_access(
        self,
        user_role: str,
        requested_action: str,
        resource_type: str
    ) -> Dict[str, Any]:
        """
        Check if user has access to perform action
        
        Args:
            user_role: User's role (patient, staff, admin)
            requested_action: Action being requested
            resource_type: Type of resource being accessed
            
        Returns:
            Access decision with details
        """
        
        # Get role permissions
        role_permissions = self.access_matrix.get(user_role, {})
        
        # Admin has full access
        if user_role == 'admin' or '*' in role_permissions.get('allowed', []):
            return {
                'allowed': True,
                'reason': 'Admin access granted'
            }
        
        # Check if action is explicitly allowed
        if resource_type in role_permissions.get('allowed', []):
            return {
                'allowed': True,
                'reason': f'{user_role} has access to {resource_type}'
            }
        
        # Check if action is explicitly denied
        if resource_type in role_permissions.get('denied', []):
            return {
                'allowed': False,
                'reason': f'{user_role} is denied access to {resource_type}'
            }
        
        # Default deny
        return {
            'allowed': False,
            'reason': f'No explicit permission for {user_role} to access {resource_type}'
        }
    
    def sanitize_output(
        self,
        content: str,
        user_role: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        Sanitize content before sending to user
        
        Args:
            content: Content to sanitize
            user_role: User's role
            tenant_id: Tenant ID
            
        Returns:
            Sanitized content with metadata
        """
        
        # Detect PHI
        phi_detected = self.detect_phi(content)
        
        # Redact based on user role
        if user_role == 'patient':
            # Patients shouldn't see other patients' PHI
            sanitized = self.redact_phi(content)
        elif user_role == 'staff':
            # Staff can see some PHI but not SSN or credit cards
            sanitized = content
            for phi_type in ['ssn', 'credit_card']:
                if phi_type in self.phi_patterns:
                    pattern = self.phi_patterns[phi_type]
                    sanitized = re.sub(pattern, f'[{phi_type.upper()}_REDACTED]', sanitized)
        else:
            # Admin sees everything
            sanitized = content
        
        # Log the sanitization
        self._log_audit(
            action='sanitize_output',
            user_role=user_role,
            tenant_id=tenant_id,
            phi_detected=len(phi_detected) > 0
        )
        
        return {
            'content': sanitized,
            'phi_detected': len(phi_detected) > 0,
            'phi_types': list(phi_detected.keys()),
            'redacted': sanitized != content
        }
    
    def validate_request(
        self,
        request_text: str,
        user_role: str
    ) -> Dict[str, Any]:
        """
        Validate if request is safe to process
        
        Args:
            request_text: Request text
            user_role: User's role
            
        Returns:
            Validation result
        """
        
        # Check for injection attempts
        injection_patterns = [
            r'(?i)drop\s+table',
            r'(?i)delete\s+from',
            r'(?i)update\s+set',
            r'(?i)<script',
            r'(?i)javascript:',
            r'(?i)eval\(',
            r'(?i)system\(',
            r'(?i)exec\('
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, request_text):
                return {
                    'valid': False,
                    'reason': 'Potential injection attack detected',
                    'threat_level': 'high'
                }
        
        # Check for unauthorized data requests
        if user_role == 'patient':
            unauthorized_patterns = [
                r'(?i)all\s+patients',
                r'(?i)other\s+patients',
                r'(?i)database',
                r'(?i)admin',
                r'(?i)system\s+config'
            ]
            
            for pattern in unauthorized_patterns:
                if re.search(pattern, request_text):
                    return {
                        'valid': False,
                        'reason': 'Unauthorized data request',
                        'threat_level': 'medium'
                    }
        
        return {
            'valid': True,
            'reason': 'Request validated successfully',
            'threat_level': 'none'
        }
    
    def _log_audit(
        self,
        action: str,
        user_role: str,
        tenant_id: str,
        **kwargs
    ):
        """
        Log audit entry
        
        Args:
            action: Action performed
            user_role: User's role
            tenant_id: Tenant ID
            **kwargs: Additional data to log
        """
        
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'user_role': user_role,
            'tenant_id': tenant_id,
            **kwargs
        }
        
        self.audit_log.append(audit_entry)
        logger.info(f"Audit: {action} by {user_role} in tenant {tenant_id}")
    
    def get_audit_log(
        self,
        limit: int = 100,
        filter_action: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries
        
        Args:
            limit: Maximum number of entries
            filter_action: Filter by action type
            
        Returns:
            List of audit entries
        """
        
        logs = self.audit_log
        
        if filter_action:
            logs = [log for log in logs if log['action'] == filter_action]
        
        return logs[-limit:]
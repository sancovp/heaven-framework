# Creating Constrained Tools with OmniTool

This guide shows how to create higher-level, domain-specific tools by using OmniTool to call other HEAVEN tools with constraints.

## The Pattern

OmniTool allows you to create specialized tools that:
1. Call other tools internally with specific constraints
2. Expose domain-specific interfaces
3. Enforce schemas and validation
4. Hide complexity from agents

## Example: ContactRegistryTool

Let's create a tool that manages a contact list by constraining RegistryTool operations to a specific registry with a specific schema.

### Step 1: Define the Contact Schema

```python
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class Contact(BaseModel):
    """Schema for a contact entry"""
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    company: Optional[str] = Field(None, description="Company name")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
```

### Step 2: Create the Constrained Tool

```python
from heaven_base.baseheaventool import BaseHeavenTool, ToolArgsSchema
from heaven_base.utils.omnitool import omnitool
import json

class ContactRegistryToolArgsSchema(ToolArgsSchema):
    arguments = {
        'operation': {
            'type': 'str',
            'description': 'Operation to perform: add_contact, get_contact, update_contact, delete_contact, list_contacts, search_by_tag',
            'required': True
        },
        'contact_id': {
            'type': 'str',
            'description': 'Unique ID for the contact (email address)',
            'required': False
        },
        'contact_data': {
            'type': 'dict',
            'description': 'Contact information (name, email, phone, company, tags)',
            'required': False
        },
        'tag': {
            'type': 'str',
            'description': 'Tag to search for',
            'required': False
        }
    }

def manage_contacts(operation: str, contact_id: str = None, 
                   contact_data: dict = None, tag: str = None) -> str:
    """
    Manage contacts using a constrained registry.
    This function uses OmniTool to call RegistryTool with specific constraints.
    """
    
    REGISTRY_NAME = "contact_registry"
    
    try:
        if operation == "add_contact":
            if not contact_data or 'email' not in contact_data:
                return "Error: contact_data with email required for add_contact"
            
            # Validate contact data against schema
            try:
                contact = Contact(**contact_data)
            except Exception as e:
                return f"Invalid contact data: {e}"
            
            # Use OmniTool to call RegistryTool
            result = omnitool('RegistryTool', parameters={
                'operation': 'set',
                'registry_name': REGISTRY_NAME,
                'key': contact.email,  # Use email as unique key
                'value': contact.dict()
            })
            
            return f"Added contact: {contact.name} ({contact.email})"
            
        elif operation == "get_contact":
            if not contact_id:
                return "Error: contact_id (email) required for get_contact"
            
            result = omnitool('RegistryTool', parameters={
                'operation': 'get',
                'registry_name': REGISTRY_NAME,
                'key': contact_id
            })
            
            if isinstance(result, str) and 'not found' in result.lower():
                return f"Contact {contact_id} not found"
            
            return json.dumps(result, indent=2)
            
        elif operation == "update_contact":
            if not contact_id or not contact_data:
                return "Error: contact_id and contact_data required for update_contact"
            
            # First get existing contact
            existing = omnitool('RegistryTool', parameters={
                'operation': 'get',
                'registry_name': REGISTRY_NAME,
                'key': contact_id
            })
            
            if isinstance(existing, str) and 'not found' in existing.lower():
                return f"Contact {contact_id} not found"
            
            # Merge with updates
            if isinstance(existing, dict):
                existing.update(contact_data)
                contact = Contact(**existing)
            else:
                return "Error retrieving existing contact"
            
            # Save updated contact
            result = omnitool('RegistryTool', parameters={
                'operation': 'set',
                'registry_name': REGISTRY_NAME,
                'key': contact_id,
                'value': contact.dict()
            })
            
            return f"Updated contact: {contact_id}"
            
        elif operation == "delete_contact":
            if not contact_id:
                return "Error: contact_id required for delete_contact"
            
            result = omnitool('RegistryTool', parameters={
                'operation': 'delete',
                'registry_name': REGISTRY_NAME,
                'key': contact_id
            })
            
            return f"Deleted contact: {contact_id}"
            
        elif operation == "list_contacts":
            # Get all keys
            keys_result = omnitool('RegistryTool', parameters={
                'operation': 'list_keys',
                'registry_name': REGISTRY_NAME
            })
            
            if not keys_result or keys_result == "[]":
                return "No contacts found"
            
            # Get all contacts
            contacts = []
            keys = json.loads(keys_result) if isinstance(keys_result, str) else keys_result
            
            for key in keys:
                contact_data = omnitool('RegistryTool', parameters={
                    'operation': 'get',
                    'registry_name': REGISTRY_NAME,
                    'key': key
                })
                if contact_data:
                    contacts.append(contact_data)
            
            return json.dumps(contacts, indent=2)
            
        elif operation == "search_by_tag":
            if not tag:
                return "Error: tag required for search_by_tag"
            
            # Get all contacts and filter by tag
            keys_result = omnitool('RegistryTool', parameters={
                'operation': 'list_keys',
                'registry_name': REGISTRY_NAME
            })
            
            if not keys_result or keys_result == "[]":
                return "No contacts found"
            
            keys = json.loads(keys_result) if isinstance(keys_result, str) else keys_result
            matching_contacts = []
            
            for key in keys:
                contact_data = omnitool('RegistryTool', parameters={
                    'operation': 'get',
                    'registry_name': REGISTRY_NAME,
                    'key': key
                })
                
                if contact_data and 'tags' in contact_data:
                    if tag in contact_data['tags']:
                        matching_contacts.append(contact_data)
            
            if not matching_contacts:
                return f"No contacts found with tag: {tag}"
            
            return json.dumps(matching_contacts, indent=2)
            
        else:
            return f"Unknown operation: {operation}"
            
    except Exception as e:
        return f"Error in ContactRegistryTool: {e}"

class ContactRegistryTool(BaseHeavenTool):
    name = "ContactRegistryTool"
    description = """Manages a contact list with structured operations.
    
    Operations:
    - add_contact: Add a new contact (requires contact_data with at least name and email)
    - get_contact: Get a specific contact by email (requires contact_id)
    - update_contact: Update an existing contact (requires contact_id and contact_data)
    - delete_contact: Remove a contact (requires contact_id)
    - list_contacts: Get all contacts
    - search_by_tag: Find contacts with a specific tag (requires tag)
    
    Contact data structure:
    - name: Full name (required)
    - email: Email address (required, used as unique ID)
    - phone: Phone number (optional)
    - company: Company name (optional)
    - tags: List of tags for categorization (optional)
    """
    func = manage_contacts
    args_schema = ContactRegistryToolArgsSchema
    is_async = False
```

## How It Works

1. **Constraint Layer**: The tool defines a specific registry name (`contact_registry`) that all operations use
2. **Schema Enforcement**: Uses Pydantic models to validate contact data
3. **OmniTool Calls**: All registry operations go through OmniTool → RegistryTool
4. **Domain Operations**: Exposes contact-specific operations instead of generic registry operations
5. **Error Handling**: Provides user-friendly error messages

## Benefits of This Pattern

### 1. Simplified Agent Interface
Instead of:
```python
# Agent has to know registry operations
"Use RegistryTool to set key 'john@example.com' in registry 'contact_registry' with value..."
```

The agent can simply:
```python
# Agent uses domain-specific operations
"Add a contact for John Doe with email john@example.com"
```

### 2. Data Validation
The tool ensures all data conforms to the Contact schema before storing.

### 3. Reusability
Once created, this tool can be used by any agent that needs contact management.

### 4. Composability
You can create multiple constrained tools for different registries:
- `TaskRegistryTool` for task management
- `ProjectRegistryTool` for project tracking
- `ConfigRegistryTool` for configuration management

## Advanced Pattern: Multi-Tool Orchestration

You can also use OmniTool to orchestrate multiple tools:

```python
def create_project_with_contacts(project_name: str, team_emails: List[str]) -> str:
    """
    Creates a project and associates team members using multiple tools.
    """
    
    # Create project using one tool
    project_result = omnitool('ProjectTool', parameters={
        'operation': 'create',
        'name': project_name
    })
    
    # Get contacts using ContactRegistryTool
    team_members = []
    for email in team_emails:
        contact = omnitool('ContactRegistryTool', parameters={
            'operation': 'get_contact',
            'contact_id': email
        })
        if contact:
            team_members.append(contact)
    
    # Associate team with project
    assoc_result = omnitool('ProjectTool', parameters={
        'operation': 'add_team',
        'project_name': project_name,
        'team': team_members
    })
    
    return f"Created project {project_name} with {len(team_members)} team members"
```

## Another Example: Essay Editing Tool

Here's the 5-paragraph essay tool concept mentioned:

```python
class EssaySchema(BaseModel):
    title: str
    introduction: str
    body_paragraph_1: str
    body_paragraph_2: str
    body_paragraph_3: str
    conclusion: str

def manage_essay(operation: str, essay_id: str = None, 
                section: str = None, content: str = None) -> str:
    """
    Manages essay editing with structured operations.
    """
    
    if operation == "create":
        # Create essay file structure
        essay_path = f"/essays/{essay_id}.json"
        
        result = omnitool('NetworkEditTool', parameters={
            'command': 'create',
            'path': essay_path,
            'file_text': json.dumps({
                'title': '',
                'introduction': '',
                'body_paragraph_1': '',
                'body_paragraph_2': '',
                'body_paragraph_3': '',
                'conclusion': ''
            }, indent=2)
        })
        
    elif operation == "update_section":
        # Update specific section
        essay_path = f"/essays/{essay_id}.json"
        
        # First read current essay
        current = omnitool('NetworkEditTool', parameters={
            'command': 'view',
            'path': essay_path,
            'command_arguments': {}
        })
        
        essay_data = json.loads(current)
        essay_data[section] = content
        
        # Validate against schema
        essay = EssaySchema(**essay_data)
        
        # Write back
        result = omnitool('NetworkEditTool', parameters={
            'command': 'str_replace',
            'path': essay_path,
            'old_str': current,
            'new_str': json.dumps(essay.dict(), indent=2)
        })
    
    # ... more operations
```

## Key Takeaways

1. **OmniTool enables tool composition** - Create higher-level tools from lower-level ones
2. **Constraints provide safety** - Limit what operations are possible
3. **Schemas ensure consistency** - Validate data at the tool level
4. **Domain-specific interfaces** - Make tools easier for agents to use
5. **This is a valid use of OmniTool** - It's programmatic tool orchestration, not violating the "code calls functions" rule

This pattern is especially powerful for:
- Creating domain-specific tool sets
- Enforcing business logic at the tool level
- Simplifying complex multi-tool workflows
- Building reusable tool compositions
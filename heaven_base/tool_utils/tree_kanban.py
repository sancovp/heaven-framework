from typing import List, Optional
from .github_kanban import KanbanBoard, Issue, construct_kanban_from_labels

def parse_tree_priority(priority_str: str) -> List[int]:
    """Parse tree notation priority into list of integers for sorting"""
    if priority_str == 'high':
        return [1]
    elif priority_str == 'medium':
        return [2] 
    elif priority_str == 'low':
        return [3]
    
    try:
        return [int(part) for part in priority_str.split('.')]
    except ValueError:
        return [999]

def get_issue_priority(issue: Issue) -> List[int]:
    """Get priority from issue labels"""
    for label in issue.labels:
        if label.startswith('priority-'):
            priority_part = label[9:]
            return parse_tree_priority(priority_part)
    return [999]

def sort_issues_by_tree_priority(issues: List[Issue]) -> List[Issue]:
    """Sort issues by tree priority"""
    return sorted(issues, key=get_issue_priority)

def construct_tree_kanban(repo: str = 'sancovp/heaven-base') -> KanbanBoard:
    """Construct kanban board with tree priority sorting"""
    board = construct_kanban_from_labels(repo)
    
    # Sort all lanes by tree priority
    board.backlog = sort_issues_by_tree_priority(board.backlog)
    board.plan = sort_issues_by_tree_priority(board.plan)
    board.build = sort_issues_by_tree_priority(board.build)
    board.measure = sort_issues_by_tree_priority(board.measure)
    board.learn = sort_issues_by_tree_priority(board.learn)
    board.blocked = sort_issues_by_tree_priority(board.blocked)
    board.archived = sort_issues_by_tree_priority(board.archived)
    
    return board

def get_issue_priority_string(issue: Issue) -> Optional[str]:
    """Get priority string from issue labels"""
    for label in issue.labels:
        if label.startswith('priority-'):
            return label[9:]
    return None

def demo_tree_kanban():
    """Demo the tree kanban system"""
    board = construct_tree_kanban()
    print("🌳 TREE KANBAN BOARD")
    print("=" * 50)
    print(f"📋 PLAN ({len(board.plan)} issues)")
    for issue in board.plan:
        priority = get_issue_priority_string(issue)
        print(f"   #{issue.number}: {issue.title[:50]} (priority: {priority or 'none'})")


def create_priority_label_if_needed(repo: str, priority: str) -> bool:
    """Create priority label if it doesn't exist"""
    import subprocess
    import json
    
    label_name = f'priority-{priority}'
    
    # Check if label exists
    try:
        cmd = f'gh api repos/{repo}/labels/{label_name}'
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return True  # Label exists
    except subprocess.CalledProcessError:
        pass  # Label doesn't exist, create it
    
    # Create label with tree-aware color coding
    depth = priority.count('.')
    colors = ['1f77b4', '2ca02c', 'd62728', 'ff7f0e', '9467bd', '8c564b', 'e377c2', '7f7f7f', 'bcbd22']
    color = colors[depth % len(colors)]
    
    try:
        cmd = f'gh api repos/{repo}/labels -f name="{label_name}" -f color="{color}" -f description="Tree priority {priority}"'
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create label {label_name}: {e}")
        return False

def set_issue_tree_priority(repo: str, issue_number: int, priority: str) -> bool:
    """Set tree priority on issue, creating label if needed"""
    import subprocess
    import json
    
    # Create label if needed
    if not create_priority_label_if_needed(repo, priority):
        return False
    
    # Remove existing priority labels
    try:
        cmd = f'gh issue view {issue_number} --repo {repo} --json labels'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        labels = json.loads(result.stdout)['labels']
        
        for label in labels:
            if label['name'].startswith('priority-'):
                subprocess.run(f'gh issue edit {issue_number} --repo {repo} --remove-label "{label["name"]}"', shell=True)
    except:
        pass
    
    # Add new priority label
    try:
        cmd = f'gh issue edit {issue_number} --repo {repo} --add-label "priority-{priority}"'
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def get_parent_priority(priority: str) -> str:
    """Get parent priority from tree notation (1.2.3 -> 1.2)"""
    parts = priority.split('.')
    return '.'.join(parts[:-1]) if len(parts) > 1 else None

def find_issue_by_priority(repo: str, target_priority: str):
    """Find issue with specific priority"""
    import subprocess
    import json
    
    try:
        # Search for issues with the priority label
        cmd = f'gh issue list --repo {repo} --label "priority-{target_priority}" --json number,labels --limit 1'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        issues = json.loads(result.stdout)
        return issues[0]['number'] if issues else None
    except:
        return None

def get_issue_status(repo: str, issue_number: int) -> str:
    """Get current status of an issue"""
    import subprocess
    import json
    
    try:
        cmd = f'gh issue view {issue_number} --repo {repo} --json labels'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        labels = json.loads(result.stdout)['labels']
        
        for label in labels:
            if label['name'].startswith('status-'):
                return label['name'][7:]  # Remove 'status-' prefix
        return 'backlog'  # Default status
    except:
        return 'backlog'

def set_issue_status(repo: str, issue_number: int, status: str) -> bool:
    """Set status label on issue"""
    import subprocess
    import json
    
    try:
        # Remove existing status labels
        cmd = f'gh issue view {issue_number} --repo {repo} --json labels'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        labels = json.loads(result.stdout)['labels']
        
        for label in labels:
            if label['name'].startswith('status-'):
                subprocess.run(f'gh issue edit {issue_number} --repo {repo} --remove-label "{label["name"]}"', shell=True)
        
        # Add new status label
        cmd = f'gh issue edit {issue_number} --repo {repo} --add-label "status-{status}"'
        subprocess.run(cmd, shell=True, check=True)
        return True
    except:
        return False

def set_issue_tree_priority_with_inheritance(repo: str, issue_number: int, priority: str) -> bool:
    """Set tree priority with automatic status inheritance from parent"""
    
    # First set the priority label
    if not set_issue_tree_priority(repo, issue_number, priority):
        return False
    
    # Check if this is a subtask (has parent)
    parent_priority = get_parent_priority(priority)
    if parent_priority:
        # Find parent issue
        parent_issue = find_issue_by_priority(repo, parent_priority)
        if parent_issue:
            # Inherit parent's status
            parent_status = get_issue_status(repo, parent_issue)
            set_issue_status(repo, issue_number, parent_status)
            print(f"Issue #{issue_number} inherited status '{parent_status}' from parent #{parent_issue}")
    
    return True

def sync_tree_statuses(repo: str, root_priority: str) -> None:
    """Sync all issues in a tree to have consistent statuses"""
    import subprocess
    import json
    
    # Get all issues with priorities starting with root_priority
    try:
        cmd = f'gh issue list --repo {repo} --json number,labels --limit 100'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        all_issues = json.loads(result.stdout)
        
        tree_issues = []
        for issue in all_issues:
            for label in issue['labels']:
                if label['name'].startswith(f'priority-{root_priority}'):
                    priority = label['name'][9:]  # Remove 'priority-' prefix
                    tree_issues.append((issue['number'], priority))
                    break
        
        # Sort by tree depth (parents first)
        tree_issues.sort(key=lambda x: x[1].count('.'))
        
        # Apply status inheritance
        for issue_num, priority in tree_issues:
            parent_priority = get_parent_priority(priority)
            if parent_priority:
                parent_issue = find_issue_by_priority(repo, parent_priority)
                if parent_issue:
                    parent_status = get_issue_status(repo, parent_issue)
                    set_issue_status(repo, issue_num, parent_status)
                    print(f"Synced #{issue_num} (priority {priority}) to status '{parent_status}'")
    
    except Exception as e:
        print(f"Error syncing tree statuses: {e}")


# Simple Priority Management - List Manipulation + Renumbering
# ===========================================================

def get_all_prioritized_issues(repo: str) -> List[dict]:
    """Get all issues with priorities, sorted by current priority."""
    import requests
    import os
    
    headers = {
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    try:
        response = requests.get(f'https://api.github.com/repos/{repo}/issues?state=open&per_page=100', headers=headers)
        response.raise_for_status()
        issues = response.json()
        
        prioritized_issues = []
        for issue in issues:
            priority = None
            priority_labels = []
            
            # Collect all priority labels
            for label in issue.get('labels', []):
                if label['name'].startswith('priority-'):
                    priority_labels.append(label['name'][9:])  # Remove 'priority-' prefix
            
            if priority_labels:
                # Prefer 0.x format labels over old format (high, medium, low, numbers)
                numeric_priorities = [p for p in priority_labels if p.startswith('0.')]
                if numeric_priorities:
                    priority = numeric_priorities[0]  # Take first 0.x priority
                else:
                    priority = priority_labels[0]  # Fall back to first priority
            
            if priority:
                prioritized_issues.append({
                    'number': issue['number'],
                    'title': issue['title'],
                    'priority': priority,
                    'priority_parsed': parse_tree_priority(priority)
                })
        
        # Sort by parsed priority
        prioritized_issues.sort(key=lambda x: x['priority_parsed'])
        return prioritized_issues
        
    except Exception as e:
        print(f"Error getting prioritized issues: {e}")
        return []

def renumber_all_priorities(issues: List[dict], repo: str):
    """Assign clean sequential priorities to list of issues, preserving tree structure."""
    import requests
    import os
    
    # Group by tree level 
    root_issues = []
    subtasks = {}  # parent_priority -> [subtasks]
    
    for issue in issues:
        priority_parts = issue['priority'].split('.')
        if len(priority_parts) == 1:
            root_issues.append(issue)
        else:
            parent_priority = '.'.join(priority_parts[:-1])
            if parent_priority not in subtasks:
                subtasks[parent_priority] = []
            subtasks[parent_priority].append(issue)
    
    # Renumber root issues
    for i, issue in enumerate(root_issues):
        new_priority = f"{i+1}"
        set_issue_tree_priority(repo, str(issue['number']), new_priority)
        issue['new_priority'] = new_priority
        
        # Renumber subtasks under this root
        old_priority = issue['priority']
        if old_priority in subtasks:
            for j, subtask in enumerate(subtasks[old_priority]):
                subtask_priority = f"{new_priority}.{j+1}"
                set_issue_tree_priority(repo, str(subtask['number']), subtask_priority)
                subtask['new_priority'] = subtask_priority

def move_issue_above(issue_id: str, target_issue_id: str, repo: str) -> str:
    """Move issue to position above target issue."""
    all_issues = get_all_prioritized_issues(repo)
    
    # Find and remove the issue we're moving
    moving_issue = None
    for i, issue in enumerate(all_issues):
        if str(issue['number']) == str(issue_id):
            moving_issue = all_issues.pop(i)
            break
    
    if not moving_issue:
        raise ValueError(f"Issue #{issue_id} not found or has no priority")
    
    # Find target position
    target_position = None
    for i, issue in enumerate(all_issues):
        if str(issue['number']) == str(target_issue_id):
            target_position = i
            break
    
    if target_position is None:
        raise ValueError(f"Target issue #{target_issue_id} not found")
    
    # Insert above target
    all_issues.insert(target_position, moving_issue)
    
    # Renumber everything
    renumber_all_priorities(all_issues, repo)
    
    # Return new priority
    new_priority = f"{target_position + 1}"
    print(f"Moved issue #{issue_id} above #{target_issue_id} → priority {new_priority}")
    return new_priority

def move_issue_below(issue_id: str, target_issue_id: str, repo: str) -> str:
    """Move issue to position below target issue."""
    all_issues = get_all_prioritized_issues(repo)
    
    # Find and remove the issue we're moving
    moving_issue = None
    for i, issue in enumerate(all_issues):
        if str(issue['number']) == str(issue_id):
            moving_issue = all_issues.pop(i)
            break
    
    if not moving_issue:
        raise ValueError(f"Issue #{issue_id} not found or has no priority")
    
    # Find target position
    target_position = None
    for i, issue in enumerate(all_issues):
        if str(issue['number']) == str(target_issue_id):
            target_position = i + 1  # Insert after target
            break
    
    if target_position is None:
        raise ValueError(f"Target issue #{target_issue_id} not found")
    
    # Insert below target
    all_issues.insert(target_position, moving_issue)
    
    # Renumber everything
    renumber_all_priorities(all_issues, repo)
    
    # Return new priority
    new_priority = f"{target_position + 1}"
    print(f"Moved issue #{issue_id} below #{target_issue_id} → priority {new_priority}")
    return new_priority

def move_issue_between(issue_id: str, above_issue_id: str, below_issue_id: str, repo: str) -> str:
    """Move issue between two other issues."""
    all_issues = get_all_prioritized_issues(repo)
    
    # Find and remove the issue we're moving
    moving_issue = None
    for i, issue in enumerate(all_issues):
        if str(issue['number']) == str(issue_id):
            moving_issue = all_issues.pop(i)
            break
    
    if not moving_issue:
        raise ValueError(f"Issue #{issue_id} not found or has no priority")
    
    # Find positions of above and below issues
    above_position = None
    below_position = None
    
    for i, issue in enumerate(all_issues):
        if str(issue['number']) == str(above_issue_id):
            above_position = i
        elif str(issue['number']) == str(below_issue_id):
            below_position = i
    
    if above_position is None:
        raise ValueError(f"Above issue #{above_issue_id} not found")
    if below_position is None:
        raise ValueError(f"Below issue #{below_issue_id} not found")
    
    if above_position >= below_position:
        raise ValueError(f"Above issue must have higher priority than below issue")
    
    # Insert between them (right after the above issue)
    insert_position = above_position + 1
    all_issues.insert(insert_position, moving_issue)
    
    # Renumber everything
    renumber_all_priorities(all_issues, repo)
    
    # Return new priority (will be determined by renumbering)
    new_priority = f"{insert_position + 1}"
    print(f"Moved issue #{issue_id} between #{above_issue_id} and #{below_issue_id} → priority {new_priority}")
    return new_priority


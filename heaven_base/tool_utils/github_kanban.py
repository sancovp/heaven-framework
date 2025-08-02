#\!/usr/bin/env python3
"""
GitHub Kanban Functions
Manages kanban board state via GitHub issue labels
"""

import subprocess
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Issue:
    number: int
    title: str
    state: str
    labels: List[str]
    assignees: List[str]
    url: str

@dataclass
class KanbanBoard:
    backlog: List[Issue]
    plan: List[Issue] 
    build: List[Issue]
    measure: List[Issue]
    learn: List[Issue]
    blocked: List[Issue]
    archived: List[Issue]

# Workflow transition rules
VALID_TRANSITIONS = {
    'backlog': ['plan', 'blocked'],
    'plan': ['build', 'blocked', 'backlog'],  # Can go back to backlog
    'build': ['measure', 'blocked', 'plan'],  # Can go back to plan
    'measure': ['learn', 'blocked', 'build'], # Can go back to build  
    'learn': ['archived', 'backlog'],  # Can spawn new work or archive
    'blocked': ['backlog', 'plan', 'build', 'measure'],  # Can unblock to any previous state
    'archived': []  # Terminal state
}

def gh_search_issues(repo: str, label: str) -> List[Issue]:
    """Search GitHub issues by label"""
    try:
        cmd = f'gh issue list --repo {repo} --label "{label}" --json number,title,state,labels,assignees,url --limit 100'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        issues_data = json.loads(result.stdout)
        
        issues = []
        for issue_data in issues_data:
            issue = Issue(
                number=issue_data['number'],
                title=issue_data['title'], 
                state=issue_data['state'],
                labels=[label['name'] for label in issue_data['labels']],
                assignees=[assignee['login'] for assignee in issue_data['assignees']], 
                url=issue_data['url']
            )
            issues.append(issue)
        return issues
    except subprocess.CalledProcessError as e:
        print(f"Error searching issues: {e.stderr}")
        return []

def get_issue_status(issue: Issue) -> Optional[str]:
    """Extract status from issue labels"""
    for label in issue.labels:
        if label.startswith('status-'):
            return label.replace('status-', '')
    return None

def construct_kanban_from_labels(repo: str = 'sancovp/heaven-base') -> KanbanBoard:
    """Construct kanban board from GitHub issue labels"""
    print(f"Constructing kanban board for {repo}...")
    
    # Get all issues with status labels
    all_issues = []
    statuses = ['backlog', 'plan', 'build', 'measure', 'learn', 'blocked', 'archived']
    
    for status in statuses:
        status_issues = gh_search_issues(repo, f'status-{status}')
        all_issues.extend(status_issues)
    
    # Organize by status
    board = KanbanBoard([], [], [], [], [], [], [])
    
    for issue in all_issues:
        status = get_issue_status(issue)
        if status == 'backlog':
            board.backlog.append(issue)
        elif status == 'plan':
            board.plan.append(issue)
        elif status == 'build':
            board.build.append(issue)
        elif status == 'measure':
            board.measure.append(issue)
        elif status == 'learn':
            board.learn.append(issue)
        elif status == 'blocked':
            board.blocked.append(issue)
        elif status == 'archived':
            board.archived.append(issue)
    
    return board

def view_lane(repo: str, lane: str) -> List[Issue]:
    """View specific kanban lane"""
    if lane not in ['backlog', 'plan', 'build', 'measure', 'learn', 'blocked', 'archived']:
        raise ValueError(f"Invalid lane: {lane}. Must be one of: backlog, plan, build, measure, learn, blocked, archived")
    
    print(f"Viewing {lane} lane for {repo}...")
    issues = gh_search_issues(repo, f'status-{lane}')
    
    print(f"\n=== {lane.upper()} ({len(issues)} issues) ===")
    for issue in issues:
        assignee_str = f" [@{', @'.join(issue.assignees)}]" if issue.assignees else ""
        print(f"#{issue.number}: {issue.title}{assignee_str}")
        print(f"  {issue.url}")
    
    return issues

def get_issue_pr_links(repo: str, issue_number: int) -> List[str]:
    """Get PR links associated with an issue"""
    try:
        # Search for PRs that reference this issue
        cmd = f'gh pr list --repo {repo} --search "#{issue_number}" --json number,title,url'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        prs_data = json.loads(result.stdout)
        return [pr['url'] for pr in prs_data]
    except subprocess.CalledProcessError as e:
        print(f"Error getting PR links: {e.stderr}")
        return []

def move_issue_to_next_status(repo: str, issue_number: int, target_status: str, pr_id: Optional[str] = None) -> bool:
    """Move issue to next status with workflow validation"""
    print(f"Moving issue #{issue_number} to {target_status}...")
    
    # Get current issue data
    try:
        cmd = f'gh issue view {issue_number} --repo {repo} --json number,labels'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        issue_data = json.loads(result.stdout)
        current_labels = [label['name'] for label in issue_data['labels']]
    except subprocess.CalledProcessError as e:
        print(f"Error getting issue data: {e.stderr}")
        return False
    
    # Find current status
    current_status = None
    for label in current_labels:
        if label.startswith('status-'):
            current_status = label.replace('status-', '')
            break
    
    if not current_status:
        print(f"No current status found for issue #{issue_number}")
        return False
    
    # Validate transition
    if target_status not in VALID_TRANSITIONS.get(current_status, []):
        print(f"Invalid transition: {current_status} -> {target_status}")
        print(f"Valid transitions from {current_status}: {VALID_TRANSITIONS.get(current_status, [])}")
        return False
    
    # Special validation: build -> measure requires PR
    if current_status == 'build' and target_status == 'measure':
        if not pr_id:
            # Check if there are any PRs linked to this issue
            pr_links = get_issue_pr_links(repo, issue_number)
            if not pr_links:
                print(f"Cannot move from build to measure without a PR. No PRs found for issue #{issue_number}")
                return False
            else:
                print(f"Found PRs for issue #{issue_number}: {pr_links}")
    
    # Remove old status label
    try:
        cmd = f'gh issue edit {issue_number} --repo {repo} --remove-label "status-{current_status}"'
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error removing old label: {e.stderr}")
    
    # Add new status label  
    try:
        cmd = f'gh issue edit {issue_number} --repo {repo} --add-label "status-{target_status}"'
        subprocess.run(cmd, shell=True, check=True)
        
        # Add comment about transition
        comment = f"🔄 **Status changed:** {current_status} → {target_status}"
        if pr_id:
            comment += f"\n📋 **PR:** {pr_id}"
        
        cmd = f'gh issue comment {issue_number} --repo {repo} --body "{comment}"'
        subprocess.run(cmd, shell=True, check=True)
        
        print(f"✅ Successfully moved issue #{issue_number} from {current_status} to {target_status}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error adding new label: {e.stderr}")
        return False

def move_issue_to_blocked(repo: str, issue_number: int, reason: str) -> bool:
    """Move issue to blocked status with required reason"""
    if not reason or reason.strip() == '':
        print("Error: Reason is required when moving issue to blocked status")
        return False
    
    print(f"Moving issue #{issue_number} to blocked with reason: {reason}")
    
    # Get current status  
    try:
        cmd = f'gh issue view {issue_number} --repo {repo} --json labels'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        issue_data = json.loads(result.stdout)
        current_labels = [label['name'] for label in issue_data['labels']]
    except subprocess.CalledProcessError as e:
        print(f"Error getting issue data: {e.stderr}")
        return False
    
    # Find current status
    current_status = None
    for label in current_labels:
        if label.startswith('status-'):
            current_status = label.replace('status-', '')
            break
    
    if current_status == 'blocked':
        print(f"Issue #{issue_number} is already blocked")
        return True
    
    # Remove old status label
    if current_status:
        try:
            cmd = f'gh issue edit {issue_number} --repo {repo} --remove-label "status-{current_status}"'
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error removing old label: {e.stderr}")
    
    # Add blocked status label
    try:
        cmd = f'gh issue edit {issue_number} --repo {repo} --add-label "status-blocked"'
        subprocess.run(cmd, shell=True, check=True)
        
        # Add comment with reason
        comment = f"🚫 **Issue blocked:** {reason}\n\n**Previous status:** {current_status or 'unknown'}"
        cmd = f'gh issue comment {issue_number} --repo {repo} --body "{comment}"'
        subprocess.run(cmd, shell=True, check=True)
        
        print(f"✅ Successfully blocked issue #{issue_number}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error blocking issue: {e.stderr}")
        return False

def print_kanban_board(board: KanbanBoard):
    """Pretty print the kanban board"""
    lanes = [
        ('BACKLOG', board.backlog),
        ('PLAN', board.plan), 
        ('BUILD', board.build),
        ('MEASURE', board.measure),
        ('LEARN', board.learn),
        ('BLOCKED', board.blocked),
        ('ARCHIVED', board.archived)
    ]
    
    print("\n" + "="*80)
    print("🏗️  HEAVEN FRAMEWORK KANBAN BOARD")
    print("="*80)
    
    for lane_name, issues in lanes:
        print(f"\n📋 {lane_name} ({len(issues)} issues)")
        print("-" * 40)
        if not issues:
            print("   (empty)")
        else:
            for issue in issues[:5]:  # Show first 5 issues
                assignee_str = f" [@{', @'.join(issue.assignees)}]" if issue.assignees else ""
                print(f"   #{issue.number}: {issue.title[:50]}{'...' if len(issue.title) > 50 else ''}{assignee_str}")
            if len(issues) > 5:
                print(f"   ... and {len(issues) - 5} more")

if __name__ == "__main__":
    # Test the functions
    print("Testing GitHub Kanban functions...")
    
    # Test kanban construction
    board = construct_kanban_from_labels()
    print_kanban_board(board)
    
    # Test lane view
    print("\n" + "="*50)
    backlog_issues = view_lane('sancovp/heaven-base', 'backlog')

def create_github_issue_with_status(
    repo: str, 
    title: str, 
    body: str, 
    status: str = 'backlog',
    priority: str = 'medium',
    labels: List[str] = None
) -> int:
    """
    Create GitHub issue with automatic status labeling for agent-driven development.
    
    Args:
        repo: Repository in format 'owner/repo'
        title: Issue title
        body: Issue description  
        status: Initial status (default: backlog)
        priority: Issue priority (high/medium/low)
        labels: Additional labels to add
    
    Returns:
        Issue number of created issue
    """
    print(f"Creating issue '{title}' in {repo} with status {status}...")
    
    # Validate status
    valid_statuses = ['backlog', 'plan', 'build', 'measure', 'learn', 'blocked', 'archived']
    if status not in valid_statuses:
        raise ValueError(f"Invalid status: {status}. Must be one of: {valid_statuses}")
    
    # Create the issue
    try:
        cmd = f'gh issue create --repo {repo} --title "{title}" --body "{body}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        
        # Extract issue number from URL
        issue_url = result.stdout.strip()
        issue_number = int(issue_url.split('/')[-1])
        
        print(f"✅ Created issue #{issue_number}: {issue_url}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating issue: {e.stderr}")
        raise
    
    # Add status label immediately
    try:
        cmd = f'gh issue edit {issue_number} --repo {repo} --add-label "status-{status}"'
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Added status-{status} label")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not add status label: {e.stderr}")
    
    # Add priority label
    try:
        cmd = f'gh issue edit {issue_number} --repo {repo} --add-label "priority-{priority}"'
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Added priority-{priority} label")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not add priority label: {e.stderr}")
    
    # Add any additional labels
    if labels:
        for label in labels:
            try:
                cmd = f'gh issue edit {issue_number} --repo {repo} --add-label "{label}"'
                subprocess.run(cmd, shell=True, check=True)
                print(f"✅ Added label: {label}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not add label {label}: {e.stderr}")
    
    # Add initial comment explaining agent creation
    try:
        comment = f"🤖 **Issue created by AI agent**\n\n**Initial Status:** {status}\n**Priority:** {priority}\n\nThis issue was automatically created and labeled by an AI agent for streamlined project management."
        cmd = f'gh issue comment {issue_number} --repo {repo} --body "{comment}"'
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Added agent creation comment")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not add comment: {e.stderr}")
    
    return issue_number

def agent_create_issue_from_idea(
    repo: str,
    idea_text: str,
    extracted_title: str = None,
    priority: str = 'medium'
) -> int:
    """
    Agent utility to create issue from idea text with smart title extraction.
    
    Args:
        repo: Repository in format 'owner/repo'  
        idea_text: Raw idea text from user
        extracted_title: Pre-extracted title (optional)
        priority: Issue priority
    
    Returns:
        Issue number of created issue
    """
    
    # Extract title if not provided
    if not extracted_title:
        # Use first line as title, clean it up
        lines = idea_text.strip().split('\n')
        title = lines[0].strip()
        # Remove markdown heading markers
        title = title.lstrip('#').strip()
        # Truncate if too long
        if len(title) > 80:
            title = title[:77] + '...'
    else:
        title = extracted_title
    
    # Format body with metadata
    body = f"""**Agent-Generated Issue from User Idea**

{idea_text}

---
**Metadata:**
- Created by: AI Agent
- Source: User idea/request
- Auto-Status: backlog (ready for planning)
- Priority: {priority}
"""
    
    return create_github_issue_with_status(
        repo=repo,
        title=title, 
        body=body,
        status='backlog',
        priority=priority,
        labels=['agent-created', 'user-idea']
    )

# Test function
def test_agent_issue_creation():
    """Test agent issue creation functionality"""
    print("Testing agent issue creation...")
    
    # Test 1: Create issue from user idea
    idea = """I think we should add a dark mode toggle to the settings page. 
Users have been asking for this and it would improve accessibility.
Could include system preference detection and manual override."""
    
    try:
        issue_num = agent_create_issue_from_idea(
            repo='sancovp/heaven-base',
            idea_text=idea,
            priority='high'
        )
        print(f"✅ Test 1 passed: Created issue #{issue_num}")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Create structured issue
    try:
        issue_num = create_github_issue_with_status(
            repo='sancovp/heaven-base',
            title='Add automated testing pipeline',
            body='Set up CI/CD pipeline with automated testing for pull requests.',
            status='plan',
            priority='high',
            labels=['enhancement', 'ci-cd']
        )
        print(f"✅ Test 2 passed: Created issue #{issue_num}")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

if __name__ == "__main__":
    # Add test to main execution
    print("\n" + "="*60)
    print("Testing Agent Issue Creation")
    print("="*60)
    test_agent_issue_creation()

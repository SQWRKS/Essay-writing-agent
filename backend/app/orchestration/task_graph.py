from typing import Dict, List, Set, Optional


class TaskNode:
    def __init__(self, task_id: str, agent_name: str, dependencies: List[str] = None):
        self.task_id = task_id
        self.agent_name = agent_name
        self.dependencies: List[str] = dependencies or []
        self.status: str = "pending"


class TaskGraph:
    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}

    def add_task(self, task_id: str, agent_name: str, dependencies: List[str] = None) -> TaskNode:
        node = TaskNode(task_id, agent_name, dependencies or [])
        self.nodes[task_id] = node
        return node

    def get_ready_tasks(self) -> List[TaskNode]:
        ready = []
        for node in self.nodes.values():
            if node.status != "pending":
                continue
            deps_done = all(
                self.nodes[dep].status == "completed"
                for dep in node.dependencies
                if dep in self.nodes
            )
            if deps_done:
                ready.append(node)
        return ready

    def mark_completed(self, task_id: str):
        if task_id in self.nodes:
            self.nodes[task_id].status = "completed"

    def mark_failed(self, task_id: str):
        if task_id in self.nodes:
            self.nodes[task_id].status = "failed"

    def topological_sort(self) -> List[str]:
        visited: Set[str] = set()
        order: List[str] = []

        def dfs(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if node:
                for dep in node.dependencies:
                    dfs(dep)
                order.append(node_id)

        for node_id in list(self.nodes.keys()):
            dfs(node_id)
        return order

    def is_complete(self) -> bool:
        return all(n.status in ("completed", "failed") for n in self.nodes.values())

    def has_failed(self) -> bool:
        return any(n.status == "failed" for n in self.nodes.values())

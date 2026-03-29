import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_get_config(client):
    resp = await client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "LLM_MODEL" in data


@pytest.mark.asyncio
async def test_update_config(client):
    resp = await client.post("/api/config", json={"LLM_TEMPERATURE": 0.5})
    assert resp.status_code == 200
    assert resp.json()["LLM_TEMPERATURE"] == 0.5


@pytest.mark.asyncio
async def test_create_project(client):
    resp = await client.post("/projects", json={"title": "Test Project", "topic": "machine learning"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Test Project"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_list_projects(client):
    await client.post("/projects", json={"title": "List Test", "topic": "AI"})
    resp = await client.get("/projects")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_get_project(client):
    create_resp = await client.post("/projects", json={"title": "Detail Test", "topic": "NLP"})
    project_id = create_resp.json()["id"]
    resp = await client.get(f"/projects/{project_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == project_id
    assert "agent_states" in data
    assert "tasks" in data


@pytest.mark.asyncio
async def test_get_project_not_found(client):
    resp = await client.get("/projects/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_run_planner_agent(client):
    create_resp = await client.post("/projects", json={"title": "Planner Test", "topic": "deep learning"})
    project_id = create_resp.json()["id"]
    resp = await client.post(
        f"/projects/{project_id}/run-agent",
        json={"agent_name": "planner", "input_data": {"topic": "deep learning"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert "output" in data


@pytest.mark.asyncio
async def test_run_unknown_agent(client):
    create_resp = await client.post("/projects", json={"title": "Bad Agent", "topic": "test"})
    project_id = create_resp.json()["id"]
    resp = await client.post(
        f"/projects/{project_id}/run-agent",
        json={"agent_name": "nonexistent_agent", "input_data": {}},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_project_tasks(client):
    create_resp = await client.post("/projects", json={"title": "Tasks Test", "topic": "robotics"})
    project_id = create_resp.json()["id"]
    await client.post(
        f"/projects/{project_id}/run-agent",
        json={"agent_name": "planner", "input_data": {"topic": "robotics"}},
    )
    resp = await client.get(f"/projects/{project_id}/tasks")
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) > 0

"""FastAPI routers.

Each module defines an APIRouter that server/__init__.py includes on the app. Endpoint bodies
call the workspace registry via server.get_workspace_rag (attribute access at call time, so the
test suite's patch is honoured) and import stateless helpers directly from their owning modules.
"""

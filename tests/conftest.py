from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture(autouse=True)
def _mock_rag_chain():
    """Тесты не поднимают реальные Qdrant и LLM: lifespan получает мок-цепочку."""
    with patch("app.main.build_rag_chain") as build:
        build.return_value = (MagicMock(), MagicMock())
        yield build


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c

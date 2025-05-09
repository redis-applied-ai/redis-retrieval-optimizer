import pytest


@pytest.fixture(scope="session")
def redis_url(redis_container):
    """
    Use the `DockerCompose` fixture to get host/port of the 'redis' service
    on container port 6379 (mapped to an ephemeral port on the host).
    """
    # host, port = redis_container.get_service_host_and_port("redis", 6379)
    # return f"redis://{host}:{port}
    return "redis://localhost:6379"

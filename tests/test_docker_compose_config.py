import yaml


def test_docker_compose_services():
    """
    Ensure docker-compose.yml defines required services for local deployment.
    """
    with open("docker-compose.yml") as f:
        cfg = yaml.safe_load(f)
    services = cfg.get("services", {})
    required = {"api", "ui", "ingest", "chart-service"}
    assert required.issubset(services.keys()), f"Missing services: {required - set(services.keys())}"

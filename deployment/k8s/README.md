# HPA Autoscaling Test (Kind)

This guide describes manual steps to test Horizontal Pod Autoscaler (HPA) for the `seeragent` deployment on a local Kind cluster.

## Prerequisites

- [Kind](https://kind.sigs.k8s.io/) installed
- `kubectl` configured to point to the Kind cluster
- Kubernetes metrics-server deployed (required for HPA)

## 1. Create a Kind cluster

```bash
kind create cluster --name waveseer-test
kubectl cluster-info --context kind-waveseer-test
```

## 2. Deploy Metrics Server

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
# Wait until metrics-server is ready
kubectl wait --for=condition=available --timeout=120s deployment/metrics-server -n kube-system
```

## 3. Deploy SeerAgent and HPA

1. Apply your existing SeerAgent deployment YAML:

   ```bash
   kubectl apply -f deployment/k8s/seeragent-deployment.yaml
   ```
2. Apply HPA manifest:

   ```bash
   kubectl apply -f deployment/k8s/hpa-autoscale.yaml
   ```

## 4. Generate Load

Simulate load to trigger autoscaling. For example, port-forward and send HTTP requests or WS messages:

```bash
kubectl port-forward svc/seeragent 8000:80
for i in {1..1000}; do curl -s http://localhost:8000/health > /dev/null & done
```

Alternatively, use `hey` or `ab`:

```bash
hey -z 30s -c 50 http://localhost:8000/ping
```

## 5. Observe HPA Behavior

```bash
# Watch pod count and metrics
kubectl get hpa -w
kubectl get pods -l app=seeragent
kubectl top pods
```

Expect the number of replicas to increase toward the configured maxReplicas (5) when CPU utilization exceeds 50%, then scale down when load subsides.

## 6. Cleanup

```bash
kubectl delete hpa seeragent-hpa
kubectl delete deployment seeragent-deployment
kind delete cluster --name waveseer-test
```

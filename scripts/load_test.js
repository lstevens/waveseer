import http from 'k6/http';
import { sleep, check } from 'k6';

export let options = {
    stages: [
        { duration: '1m', target: 50 },
        { duration: '2m', target: 50 },
        { duration: '1m', target: 0 },
    ],
    rps: 50,
};

export default function () {
    const res = http.get('http://localhost:9000/health');
    check(res, { 'status is 200': (r) => r.status === 200 });
    sleep(1);
}

#!/usr/bin/env python

import argparse
import pathlib
import requests
import pandas as pd
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load testing')

    parser.add_argument('-u', '--baseUrl', dest='base_url', type=str, required=True, \
        help='Base URL to the SAM service')

    parser.add_argument('-w', '--workers', dest='num_workers', type=int, required=True, default=1, \
        help='Number of concurrent workers that hit the API')

    parser.add_argument('-r', '--requests', dest='num_requests', type=int, default=10, \
        help='Number of requests that each worker should send')

    parser.add_argument('-i', '--input', dest='input_image', type=str, default="em1.png", \
        help='Image file used as test input')
    
    parser.add_argument('--describe', action=argparse.BooleanOptionalAction, \
        help="Print statistics using Pandas describe() format instead of tab-delimited")

    args = parser.parse_args()
    input = pathlib.Path(args.input_image)

    def get_embedding_response(request_id, purge_pending):
        url = f"{args.base_url}/embedded_model"
        files = {
            'image': (input.name, open(input.absolute(), 'rb'), 'image/png', {'Expires': '0'}),
            'session_id': (None, '123435'),
            'purge_pending': (None, purge_pending)
        }
        r = requests.post(url, files=files)
        return request_id, r

    def test_SAM(worker_id):
        times = []
        futures = []
        with ThreadPoolExecutor(max_workers=args.num_requests) as executor:

            for request_id in range(args.num_requests):
                print(f"Worker {worker_id} - submitting request {request_id}")
                futures.append(executor.submit(get_embedding_response, request_id, True))
        
            for future in futures:
                request_id, r = future.result()
                if r.status_code == 499:
                    print(f"Worker {worker_id} - request {request_id} was purged")
                elif r.status_code == 200:
                    secs = r.elapsed.total_seconds()
                    print(f"Worker {worker_id} - request {request_id} completed in {secs} seconds")
                    times.append(secs)
                else:
                    print(f"Worker {worker_id} - request {request_id} FAILED with code {r.status_code}")

            print(f"Worker {worker_id} - shutting down")
            executor.shutdown(wait=True, cancel_futures=False)

        return times
        

    with Pool(args.num_workers) as p:
        series = []
        elapsed_times = p.map(test_SAM, range(args.num_workers))
        for times in elapsed_times:
            t = pd.Series(times)
            series.append(t)

        all = pd.concat(series)

        if args.describe:
            print(all.describe())
        else:
            print("Workers\tRequests\tMean\tStd\tMin\tP25\tP50\tP75\tMax")
            print(f"{args.num_workers}\t{args.num_requests}\t{all.mean()}\t{all.std()}\t{all.min()}\t{all.quantile(0.25)}\t{all.quantile(0.5)}\t{all.quantile(0.75)}\t{all.max()}")

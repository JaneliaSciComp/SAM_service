#!/usr/bin/env python
#
# This program allows testing of the "cancel" feature of the SAM Service at scale. 
# It spins up N worker processes and each one uses T threads to hit the service
# with concurrent cancelled requests.
# 

import time
import argparse
import pathlib
import requests
import pandas as pd
from multiprocessing import Pool
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

    def get_embedding_response(session_id, request_id, cancel_pending):
        url = f"{args.base_url}/embedded_model"
        files = {
            'image': (input.name, open(input.absolute(), 'rb'), 'image/png', {'Expires': '0'}),
            'session_id': (None, session_id),
            'cancel_pending': (None, cancel_pending)
        }
        r = requests.post(url, files=files)
        return request_id, r

    def sam_client(worker_id):

        # Each client needs a unique session id
        r = requests.get(f"{args.base_url}/new_session_id")
        if r.status_code != 200:
            raise Exception(f"Could not get a session id, service returned {r.status_code}")
        session_id = r.text
        
        times = []
        futures = []
        with ThreadPoolExecutor(max_workers=args.num_requests) as executor:

            for request_id in range(args.num_requests):
                print(f"Session {session_id} - submitting request {request_id}")
                future = executor.submit(get_embedding_response, session_id, request_id, True)
                futures.append(future)
                time.sleep(0.1)
        
            for future in futures:
                request_id, r = future.result()
                cancel = " (CANCELLED)" if "Cancelled-By-Client" in r.headers else ""
                
                if r.status_code == 499:
                    print(f"Session {session_id} - request {request_id}{cancel} returned 499")

                elif r.status_code == 200:
                    secs = r.elapsed.total_seconds()
                    print(f"Session {session_id} - request {request_id}{cancel} completed in {secs}s")
                    times.append(secs)

                else:
                    print(f"Session {session_id} - request {request_id} FAILED ({r.status_code})")

            print(f"Session {session_id} - shutting down")
            executor.shutdown(wait=True, cancel_futures=False)

        return times
        

    with Pool(args.num_workers) as p:
        series = []
        elapsed_times = p.map(sam_client, range(args.num_workers))
        for times in elapsed_times:
            t = pd.Series(times)
            series.append(t)

        all = pd.concat(series)

        if args.describe:
            print(all.describe())
        else:
            print("Workers\tRequests\tMean\tStd\tMin\tP25\tP50\tP75\tMax")
            print(f"{args.num_workers}\t{args.num_requests}\t{all.mean()}\t{all.std()}\t{all.min()}\t{all.quantile(0.25)}\t{all.quantile(0.5)}\t{all.quantile(0.75)}\t{all.max()}")

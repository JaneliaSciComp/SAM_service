#!/usr/bin/env python
#
# This program allows basic load testing of the SAM service by spinning up 
# N processes (i.e. clients) which each hit the service with R requests serially. 
#

import argparse
import pathlib
import requests
import pandas as pd
from multiprocessing import Pool

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

    def test_SAM(worker_id):
        times = []
        for _ in range(args.num_requests):
            url = f"{args.base_url}/embedded_model"
            files = {'image': (input.name, open(input.absolute(), 'rb'), 'image/png', {'Expires': '0'})}
            r = requests.post(url, files=files)
            assert r.status_code==200
            times.append(r.elapsed.total_seconds())
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

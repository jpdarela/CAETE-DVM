from isimip_client.client import ISIMIPClient

client = ISIMIPClient()


# S -21.500000
# N 10.500000 
# W -80.000000
# E -43.000000

BBOX = [-21.50, 10.50, -80, -43] # [south, north, west, east]

# Get the main repository results from a query
response = client.datasets(query='20crv3-era5 historical pr')

# Download the dataset files from the query response
def dowload_dataset(query_response):
    results = query_response['results']
    for result in results:
        scen = result["specifiers"]['climate_scenario']
        print(scen)
        files = result['files']
        for f in files:
            # print(f.keys())
            url = f['path']
            print(url)
            # ask = client.mask(url, bbox=BBOX)
            # client.download(ask['file_url'], path='downloads', validate=False, extract=True)
        return url

if __name__ == "__main__":
    f = dowload_dataset(response)
    ask = client.mask(f, bbox=BBOX)
    client.download(ask['file_url'])

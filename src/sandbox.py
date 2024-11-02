from worker import worker

if __name__ == "__main__":
    region = worker.load_state_zstd("./pan_amazon_hist_result.psz")


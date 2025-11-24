import separate_audio, load_datasets
#from source.pipeline import load_datasets

def main():

    # Load datasets
    load_datasets.main()

    # Separate audio
    separate_audio.main()
   

if __name__ == '__main__':
  main()
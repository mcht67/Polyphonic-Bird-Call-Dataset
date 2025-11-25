import load_datasets, separate_audio, analyze_audio
#from source.pipeline import load_datasets

def main():

    # Load datasets
    #load_datasets.main()

    # Separate audio
    separate_audio.main()

    # Analyze audio with birdnetlib
    analyze_audio.main()
   

if __name__ == '__main__':
  main()
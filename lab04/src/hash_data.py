import hashlib

def main():
    filename = "data/adult_raw.csv"

    sha256 = hashlib.sha256()

    with open(filename, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    result = sha256.hexdigest()

    print("Dataset:", filename)
    print("SHA256:", result)

    with open("lab04/results/dataset_hash.txt", "w") as file:
        file.write(result + "\n")

    print("Hash saved successfully!")

if __name__ == "__main__":
    main()
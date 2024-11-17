import modal

stub = modal.Stub("H100 test")


@stub.function(gpu="h100")
def square(x):
    print("This code is running on a remote worker!")
    return x**2


@stub.local_entrypoint()
def main():
    print("the square is", square.remote(42))

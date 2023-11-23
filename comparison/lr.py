import matplotlib.pyplot as plt


def learning_rate_schedule(step: int, model_size: int, factor: float, warmup: int) -> float:
    # we have to default the step to 1 for LambdaLR function
    # to avoid zero raising to negative power.
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def main():
    lr = []
    steps = range(33076)
    for s in steps:
        lr.append(learning_rate_schedule(s, 256, 1, 4000))

    plt.plot(lr)
    plt.show()
    print(min(lr))
    print(max(lr))


if __name__ == "__main__":
    main()

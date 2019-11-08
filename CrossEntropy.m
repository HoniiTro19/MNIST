function loss = CrossEntropy(y, tag)
    loss = -sum(tag.*log(y));
from typing import List, Dict
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt 


def get_oov_rate(train:List, test:List):
    vocabulary = set(train)
    unseen = 0
    for test_token in test:   
        if test_token not in vocabulary:
            unseen += 1
    oov = unseen / len(test) 
    return oov

 
def get_oov_rates(train:List, test:List, sizes: List) -> List:
    freq = Counter(train)
    oov_list=[]  
    # sizes = np.arange(800, 2600+1, 200)
    for vocsize in sizes:
        vocabulary_freq = freq.most_common(vocsize)
        vocabulary = [a[0] for a in vocabulary_freq]
        unseen = 0
        for test_token in test:   
            if test_token not in vocabulary:
                unseen += 1
        oov = unseen / len(test)  
        oov_list.append(oov)
    return oov_list


def plot_oov_rates(oov_rates:List, x: List, title) -> None:
    plt.figure(figsize = (15, 8))
    plt.loglog(x, oov_rates)
    plt.xlabel("Vocabulary Sizes")
    plt.ylabel("OOV Rates")
    plt.title(title)
    
    plt.show()
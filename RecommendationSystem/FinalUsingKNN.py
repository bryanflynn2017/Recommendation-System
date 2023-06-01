from knn_from_scratch import knn, euclidean_distance
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

def categoryRecommend(names, k_recommendations):
    data_set = []
    with open('recommended_item_data.csv', 'r') as md:

        next(md)

        for line in md.readlines():
            data_row = line.strip().split(',')
            data_set.append(data_row)

    recommendations = []
    for row in data_set:
        data_row = list(map(float, row[1:]))
        recommendations.append(data_row)

    recommendation_indices, _ = knn(
        recommendations, names, k=k_recommendations,
        distance_fn=euclidean_distance, choice_fn=lambda x: None
    )

    recommendations2 = []
    for _, index in recommendation_indices:
        recommendations2.append(data_set[index])

    return recommendations2

if __name__ == '__main__':
    print('(0-5) How often do you travel?')
    a = input()
    print('(0-5) Are you good with technology')
    b = input()
    print('(0-5) Do you enjoy art?')
    c = input()
    print('(0-5) Do you enjoy cooking?')
    d = input()
    print('(0-5) Are you invested in music?')
    e = input()
    print('(0-5) Are you a DIY person?')
    f = input()
    print('(0-5) Do you like pets?')
    g = input()
    print('(0-5) Are you in to fashion?')
    h = input()
    print('(0-5) Does history intrigue you?')
    i = input()
    print('(0-5) Do you like to be neat?')
    j = input()
    print('(0-5) Do you play video games?')
    k = input()
    print('(0-5) Do you read?')
    l = input()
    print('(0-5) Do you like cars?')
    m = input()
    print('(0-5) Are you a sports fan?')
    n = input()
    print('(0-5) Do you spend time around an office?')
    o = input()
    print('(0-5) Are you a gift giver?')
    p = input()
    print('(0-5) Are you decorative?')
    q = input()
    print('')
    the_post = [float(a), float(b), float(c), float(d), float(e), float(f), float(g), float(h), float(i), float(j), float(k), float(l), float(m), float(n), float(o), float(p), float(q)] # feature vector for The Post
    finalRecommeded = categoryRecommend(names=the_post, k_recommendations=4)

    originList = the_post
    predictList = categoryRecommend(names=the_post, k_recommendations=1)[0]
    predictList.pop(0)
    
    for i in range(0, len(predictList)):
        predictList[i] = int(predictList[i])
        
    
    x = pd.Series(originList, name='Answer Data')
    y = pd.Series(predictList, name='Best Match')
    table1 = pd.crosstab(x, y, margins=True)
    table2 = table1 / table1.sum(axis=1)
    print()
    print(">ERROR REPORT:")
    print(table2)
    print()
    
    def answer_plot(table2, title='Confusion matrix', cmap=plt.cm.gray_r):
        plt.matshow(table2, cmap=cmap)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(table2.columns))
        plt.xticks(tick_marks, table2.columns, rotation=45)
        plt.yticks(tick_marks, table2.index)
        plt.ylabel(table2.index.name)
        plt.xlabel(table2.columns.name)

    
    print()
    print('RECOMMENDED PRODUCTS')
    print(">RESULTS:")
    for recommendation in finalRecommeded:
        print(recommendation[0])
        
    table2 = pd.crosstab(x, y)
    answer_plot(table2)
    plt.show()
    
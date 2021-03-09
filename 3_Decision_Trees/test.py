import numpy
import operator

def estimate_gini_impurity(feature_values, threshold, labels, polarity): 
    
    # YOUR CODE HERE
    gini_impurity = 0
    
    boolean_satisfy = polarity(feature_values, threshold)
    
    num_satisfy = sum( boolean_satisfy )
    if num_satisfy==0:
        num_satisfy = numpy.finfo(float).eps
    
    num_positive = sum( labels[boolean_satisfy] == +1 )
    num_negative = sum( labels[boolean_satisfy] == -1 )
    
    p1 = num_positive / num_satisfy
    p2 = num_negative / num_satisfy
    
    gini_impurity = 1 - p1**2 - p2**2

    return gini_impurity

def estimate_gini_impurity_expectation(feature_values, threshold, labels):
    # compute P(F>T) and P(F<=T)
    p_below = sum(feature_values <= threshold) / len(feature_values)
    p_above = sum(feature_values >  threshold) / len(feature_values)
    
    # compute gini impurities
    g_below = estimate_gini_impurity(feature_values, threshold, labels, operator.le)
    g_above = estimate_gini_impurity(feature_values, threshold, labels, operator.gt)

    # compute expectation
    expectation = p_below*g_below + p_above*g_above

    return expectation

feature_values = numpy.array([1,2,3,4,5,6,7,8])
labels = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
threshold = 1
result = estimate_gini_impurity_expectation(feature_values, threshold, labels)



X = numpy.random.randint(1, 5, (8,3))
y = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
thresholds = numpy.random.randint(2, 100, (3,3))

print(thresholds)
best_threshold, best_feature = numpy.unravel_index(numpy.argmin(thresholds, axis=None), thresholds.shape)

print(best_threshold)
print(best_feature)

# print(result)
# for threshold in range(0,8): 
#     print("%.5f" % estimate_gini_impurity(feature_values, threshold, labels, operator.gt))
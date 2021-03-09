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


feature_values = numpy.array([1,2,3,4,5,6,7,8])
labels = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
threshold = 0
result = estimate_gini_impurity(feature_values, threshold, labels, operator.le)
print(result)
# for threshold in range(0,8): 
#     print("%.5f" % estimate_gini_impurity(feature_values, threshold, labels, operator.gt))
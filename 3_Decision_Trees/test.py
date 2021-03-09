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

# q1
feature_values = numpy.array([12,4,5,6,7,8,9,11])
labels = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
print( estimate_gini_impurity(feature_values, 10, labels, operator.gt) )

# q2
feature_values = numpy.array([12,4,5,6,7,8,9,11])
labels = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
print( estimate_gini_impurity(feature_values, 5, labels, operator.le) )

# q3
feature_values = numpy.array([12,4,5,6,7,8,9,11])
labels = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
print( estimate_gini_impurity_expectation(feature_values, 5, labels) )
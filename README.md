# KNN
KNN Using Majority Voting and Distance Weighted Method From Scratch

K- Nearest Neighbors Algorithm classify the Hand Written Letter in to their respective Groups.
HandWrittenLetters.txt is the data provided for classificaton.

letter_To_digit_Convert(data) function is converting the letter to its digit value.
eg A to 1 and B to 2, etc.

pickDataClass(arr1) function picks only the given class label we want to classify in to the groups.

splitData2TestTrain(number_per_class,test_instance) function splits the data into training and test instance.
Arguments passed is number of class we want to classify and number of data we want in the test instance.

calculateEuclideanDistance(X_test1) function is used to claculate the Euclidean Distance between each test point with every data point and store the distance value in distance[] matrix.

majorityVoting() function : A case is classified by a majority vote of its neighbors, with the case being assigned to the class most common amongst its K nearest neighbors measured by a distance function. 

Distance-Weighted Voting 
In the majority voting approach, every neighbor has the same impact on the classification. This makes the algorithm sensitive to the choice of k, One way to reduce the impact of k is to weight the influence of each nearest neighbor xi according to its distance: wi = 1/d(x, xi)2. As a result, training examples that are located far away from z have a weaker impact on the classification compared to those that are located close to z.

Run the whole code as it is. With python file and HandWrittenLetters.txt being in the same folder.

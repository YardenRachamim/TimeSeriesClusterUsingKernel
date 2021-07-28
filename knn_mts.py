def knn_for_mts (  testK, knn_k)   : # testK m x n , knn_k number of neighbors to return 
                                       
         # returns m x knn_k , indexes . For each test in testK , returns indexes to its knn_k nearest neighbours
         
    knn_row_count=0 
    knn_mat=np.zeros ((testK.shape[0],knn_k))
    
    for row in testK  :    # for each test MTS, consider its similarities with all train MTS
        
        knn_mat[knn_row_count,:]=np.argsort( - row)[0:knn_k]   # - for descending order
        knn_row_count+=1
        
    return knn_mat  #  an m x knn_k , each column
         
  

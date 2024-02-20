## `matlab code implementation by thaimeuu`

- list of functions (16):
  - demo_dsetdp()
  - find_nearest_point(points) -> nearest_point
  - cover_label_point_nearest(img, labels, point_crack, name) -> img_out_noslid
  - calculate_metrics(ground_truth, predicted_image) -> [precision, recall, f1]
  - Accuracy1(GT, seg) -> [F, pr, rc]
  - slidingWindowDensity(image,label) -> new_matrix
  - dset_dp_auto(descr,nsample,th_std,flag_tsne) -> [rate_nmi,rate_acc,label_c]
  - dsetpp_extend_dp(sima,dima,nsample) -> label
  - cluster_extend_dp(nneigh,dima,label,num_dsets) -> label
  - find_sigma(dima,th_std) -> sigma
  - find_delta(rho,dist) -> [delta,ordrho,nneigh]
  - clusterdata_load(idx,flag_tsne) -> [descr label fname name_data]
  - indyn(sima,x,toll) -> x
  - selectPureStrategy(x,r) -> [i]
  - nmi(x, y) -> z
  - label2accuracy(label_c,label_t) -> rate
% This is the code of the DSet-DP algorithm proposed in
% Jian Hou, Huaqiang Yuan, Marcello Pelillo. Towards Parameter-Free Clustering for Real-World Data. Pattern Recognition, vol. 134, 2023.

% the original code was modified by Le Cong Hieu
% This version is modified by me (comment out Hieu's code and add new code)

function demo_dsetdp()
    % try running the code belows inside command window line by line for visualization
    path="D:\HRG\Crack-Detection\ground_truth";
    path_gt="D:\HRG\Crack-Detection\segmentation";

    addpath(path);
    allFiles=dir(path);
    allNames={allFiles.name};
    allNames(1:2)=[];
    FileName=cellstr(allNames);
    
    %I0 = imread('F:\CRACK\skelaton\DBscan_OPTICS\skeleton_hcn_5x5\1079.png');
    H=[];
   
    flag_tsne=1;
    nsample=0.036;  % epsilon=3.6%
    th_std=0.1;  % STD=0.1
    pr=[];  % precision
    rc=[];  % recall
    f11=[];  % f1
    
    for k = 1:length(FileName)
        val_name_temp = FileName{k};  % '0002-2.png'
        val_name_temp1 = fullfile(path,val_name_temp);  % "D:\HRG\Crack-Detection\ground_truth\0002-2.png"
        val_name_temp_gt = fullfile(path_gt,val_name_temp);  % "D:\HRG\Crack-Detection\segmentation\0002-2.png"
        
        % img is an image that needs clustering
        % ground_truth is used for accuracy quantification
        img = imread(val_name_temp1);
        ground_truth = imread(val_name_temp_gt);
        
        [x,y] = find(img>0);  % x.shape, y.shape = (n, 1), (n, 1)
        point_crack = [x,y];  % point_crack.shape = (n, 2)
        descr=[x,y];

        % nmi, accuracy
        [rate_nmi,rate_acc,label_c] = dset_dp_auto(descr,nsample,th_std,flag_tsne);

        % create a grouped scatterplot, each group has a color
        gscatter(descr(:,1),descr(:,2),label_c);

        img_out_noslid = cover_label_point_nearest(img, label_c, point_crack, 'img');
      
        new_matrix = slidingWindowDensity(img,label_c);

        path_save = fullfile("D:\HRG\SVNCKH 3-2024\code availability\DSet-DP\dp-dsets-result-thaimeuu",val_name_temp);
        imwrite(img_out_noslid,path_save)

        % predicted_image = imread("D:\HRG\SVNCKH 3-2024\code availability\DSet-DP\imgwtf.png");
        % [precision, recall, f1] = calculate_metrics(ground_truth, predicted_image)
        
    end

end

function nearest_point = find_nearest_point(points)
    % returns nearest point from points' center.

    % center's coordinate
    center = mean(points);
    
    % distances from points to center
    distances = sqrt(sum((points - center).^2, 2));
    
    % returns nearest point
    [~, nearest_index] = min(distances);
    nearest_point = points(nearest_index, :);
end

function img_out_noslid = cover_label_point_nearest(img, labels, point_crack, name)
    imshow(img);
    axis off;
    point = unique(labels);
    img_out_noslid = zeros(size(img));
    hold on;

    for i = 1:numel(point)
        cluster_i = point_crack(labels == point(i), :);
        index_center = find_nearest_point(cluster_i);
        scatter(index_center(2), index_center(1), 5, 'red', 'filled');
        img_out_noslid(index_center(1), index_center(2), :) = 1;
    end

    hold off;
    % saveas(gcf, [name, '_slineding.png'], 'png');
    saveas(gcf, [name, 'thai-test.png'], 'png');
    close;
end

function [precision, recall, f1] = calculate_metrics(ground_truth, predicted_image)
    % convert to binary image if needed
    ground_truth=rgb2gray(ground_truth);
    gt_cp = zeros(size(ground_truth));

    % red dot's gray value = 76.245
    gt_cp(ground_truth == 76) = 1;

    % labeling connected components in a binary image
    labels = bwlabel(gt_cp);
    numclasses = (unique(labels));
    gt_cp = zeros(size(ground_truth));
    for k = 1:numclasses
        [x,y]= find(labels == k);
        point_crack=[x,y]
        %gt_cp(round(mean(point_crack(:,1))), round(mean(point_crack(:,2)))) = 1;

        % cluster center
        gt_cp(round(mean(point_crack(:,2))), round(mean(point_crack(:,1))))=1;
    end

    gtBinary = imbinarize(gt_cp);

    segBinary = imbinarize(predicted_image);
    unique(segBinary);

   % Calculate true positives, false positives, and false negatives
    truePositives = sum(gtBinary(:) & segBinary(:))
    falsePositives = sum(~gtBinary(:) & segBinary(:))
    falseNegatives = sum(gtBinary(:) & ~segBinary(:))

    % Calculate precision, recall, and F1 score
    precision = truePositives / (truePositives + falsePositives);
    recall = truePositives / (truePositives + falseNegatives);
    f1 = 2 * (precision * recall) / (precision + recall);
end

function [F, pr, rc] = Accuracy1(GT, seg)
    r = [];
    p = [];
    F = 0;
    
    GT(GT > 0) = 1;
    GT = GT(:);
    
    seg(seg > 0) = 1;
    seg = seg(:);
    
    CM = confusionmat(GT, seg);
    c = size(CM);
    
    for i = 1:c(1)
        if sum(CM(i, :)) == 0
            r(i) = 0;
        else
            r(i) = CM(i, i) / sum(CM(i, :));
        end
        
        if sum(CM(:, i)) == 0
            p(i) = 0;
        else
            p(i) = CM(i, i) / sum(CM(:, i));
        end
    end
    
    F = 2 * (mean(r) * mean(p)) / (mean(p) + mean(r));
    pr = mean(p);
    rc = mean(r);
end

function new_matrix = slidingWindowDensity(image,label)
% Tính density trên anh bang phuong pháp sliding windows 3x3 voi vi trí 
% trung tâm sliding ph?i có giá tr?
% Inputs:
%   - image: anh dau vào, có the là anh RGB hoac grayscale
%   - centerVal: giá tr? c?a pixel t?i v? trí trung tâm sliding
% Outputs:
%   - density: ma tran density ?ng v?i t?ng v? trí sliding trên ?nh ??u vào

% Chuy?n ??i ?nh sang grayscale n?u là ?nh RGB
if size(image, 3) == 3
    image = rgb2gray(image);
end
[x,y] = find(image>0);
point_crack=[x y];
% T?o ma tr?n density v?i kích th??c b?ng kích th??c c?a ?nh ??u vào
density = zeros(size(image));
new_matrix=zeros(size(image));
% L?p qua t?ng pixel trên ?nh
for label_i = unique(label)
    matrix_label=point_crack(label==label_i,:);
    matrix_in= zeros(max(matrix_label(:,1))+1,max(matrix_label(:,2))+1);
    % Tìm các ch? m?c t??ng ?ng v?i các ?i?m trong ma tr?n to? ??
    row_idx = round(matrix_label(:,1));
    col_idx = round(matrix_label(:,2));
    idx = sub2ind(size(matrix_in), row_idx, col_idx);

    % Thay ??i các giá tr? t?i các ch? m?c tìm ???c thành 1
    matrix_in(idx) = 1;
   
    for i =min(matrix_label(:,1)): max(matrix_label(:,1))-1
        for j = min(matrix_label(:,2)): max(matrix_label(:,2))-1
            % L?y ma tr?n 3x3 t?i v? trí sliding hi?n t?i
            window = matrix_in(i-1:i+1, j-1:j+1);
            if sum(window(:))>3 && window(2,2)==1
                sum(window(:));
                new_matrix(i,j)=sum(window(:));
            end
        end
    end
end
end


function [rate_nmi,rate_acc,label_c]=dset_dp_auto(descr,nsample,th_std,flag_tsne)

    ndataset=40;
    rate_nmi=zeros(1,ndataset);
    rate_acc=zeros(1,ndataset);
    
    
    % img = imread("D:\pix2pixHD\code khang\classfication\data\gt\0002-2.png");
    % [x,y] = find(img>0);
    % descr=[x y];
        %[descr label_t]=clusterdata_load(i,flag_tsne);
        %build sima
        dima0=pdist(descr,'euclidean');
        dima=squareform(dima0);
        d_mean=mean2(dima);
        
        sigma=find_sigma(dima,th_std);
        sima=exp(-dima/(d_mean*sigma));
        
        %do clustering
        label_c=dsetpp_extend_dp(sima,dima,nsample);
        %B = nlfilter(A, [3 3], @sliding_window_density);
      
%         unique(label_c)
%         size(label_c)
%         size(descr)
%         gscatter(descr(:,1),descr(:,2),label_c);
%         
        %fprintf('%d: %f %f\n',i,res_nmi,res_accuracy);
    

end

function label=dsetpp_extend_dp(sima,dima,nsample)

    toll=1e-4;
    ndata=size(sima,1);
    label=zeros(1,ndata);
    nsample=max(round(ndata*nsample),5);

    %dp data
    rho=zeros(1,ndata);
    for i=1:ndata
        vec=sima(i,:);
        vec1=sort(vec,'descend');
        vec_descend=vec1(2:nsample+1);
        rho(i)=mean(vec_descend);
    end

    [~,~,nneigh]=find_delta(rho,dima);
    
    min_size=3;
    th_size=min_size+1;             %the minimum size of a cluster
            
    %dset initialization
    for i=1:ndata
        sima(i,i)=0;
    end
    x=ones(ndata,1)/ndata;
    
    %start clustering
    num_dsets=0;
    while 1>0
        if sum(label==0)<5
            break;
        end

        %dset extraction
        x=indyn(sima,x,toll);
        idx_dset=find(x>0);

        if length(idx_dset)<th_size
            break;
        end

        num_dsets=num_dsets+1;
        label(idx_dset)=num_dsets;
        
        %expansion by dp
        label=cluster_extend_dp(nneigh,dima,label,num_dsets);
        
        %post-processing
        idx=label>0;
        sima(idx,:)=0;
        sima(:,idx)=0;

        idx_ndset=find(label==0);
        num_ndset=length(idx_ndset);
        x=zeros(ndata,1);
        x(idx_ndset)=1/num_ndset;
    end
    
end

function label=cluster_extend_dp(nneigh,dima,label,num_dsets)

    %start extension
    while 1>0
        idx_ndset=find(label==0);
        idx_dset=find(label==num_dsets);
        sub_dima=dima(idx_ndset,idx_dset);
        [~,idx_min]=min(sub_dima,[],1);
        idx_out=idx_ndset(idx_min);
        idx_out=unique(idx_out);

        flag=0;
        for i=idx_out
            idx_neigh=nneigh(i);
                        
            if idx_neigh==0
                continue;
            end
            
            if label(idx_neigh)==num_dsets
                label(i)=num_dsets;
                flag=1;
            end
        end
        
        if flag==0
            break;
        end
    end
    
end

function sigma=find_sigma(dima,th_std)

    dmean=mean(dima(:));

    sigma=1;
    for sigma0=1:10
        sima=exp(-dima/(dmean*sigma0));
        
        tri=triu(sima,1);
        v_tri=tri(:);
        v_sima=v_tri(v_tri>0);
        st=std(v_sima);
        
        if st<th_std
            sigma=sigma0;
            break;
        end
    end

end

function [delta,ordrho,nneigh]=find_delta(rho,dist)

    ND=length(rho);
    delta=zeros(1,ND);
    nneigh=zeros(1,ND);
    
    maxd=max(max(dist));
    
    [~,ordrho]=sort(rho,'descend');
    delta(ordrho(1))=-1;
    nneigh(ordrho(1))=0;
    
    for i=2:ND
        delta(ordrho(i))=maxd+1;
        for j=1:i-1
            if(dist(ordrho(i),ordrho(j))<delta(ordrho(i)))
                delta(ordrho(i))=dist(ordrho(i),ordrho(j));
                nneigh(ordrho(i))=ordrho(j);
            end
        end
    end
    delta(ordrho(1))=max(delta(:));
    
end

function [descr label fname name_data]=clusterdata_load(idx,flag_tsne)

    name_data=cell(1,120);
    name_data(1)={'d31.txt'};            %3100 * 2 * 31, gaussian
    name_data(2)={'r15.txt'};            %600 * 2 * 15, gaussian
    name_data(3)={'unbalance.txt'};     %6500 * 2 * 8, gaussian, cluster size
    name_data(4)={'varydensity.txt'};   %150 * 2 * 3, gaussian, density
    name_data(5)={'s1.txt'};            %5000 * 2 * 15, gaussian, overlap
    name_data(6)={'s2.txt'};            %5000 * 2 * 15, gaussian, overlap
    name_data(7)={'a1.txt'};            %3000 * 2 * 20, gaussian, overlap
    name_data(8)={'a2.txt'};            %5250 * 2 * 35, gaussian, overlap
    name_data(9)={'a3.txt'};            %7500 * 2 * 50, gaussian, overlap
    name_data(10)={'dim032.txt'};        %1024 * 32 * 16, gaussian, dimension
    name_data(11)={'dim064.txt'};        %1024 * 64 * 16, gaussian, dimension
    name_data(12)={'dim128.txt'};        %1024 * 128 * 16, gaussian, dimension
    name_data(13)={'dim256.txt'};        %1024 * 256 * 16, gaussian, dimension
    name_data(14)={'dim512.txt'};        %1024 * 512 * 16, gaussian, dimension
    name_data(15)={'dim1024.txt'};       %1024 * 1024 * 16, gaussian, dimension
    name_data(16)={'spread-2-10.txt'};   %1000 * 2 * 10
    name_data(17)={'spread-10-20.txt'};  %2000 * 10 * 20
    name_data(18)={'spread-20-35.txt'};  %3500 * 20 * 35
    name_data(19)={'spread-35-2.txt'};   %200 * 35 * 2
    name_data(20)={'spread-50-50.txt'};  %5000 * 50 * 50
    
    name_data(21)={'thyroid.txt'};        %215 * 5 * 3, uci
    name_data(22)={'wine.txt'};           %178 * 13 * 3, uci
    name_data(23)={'iris.txt'};           %150 * 4 * 3, uci
    name_data(24)={'glass.txt'};          %214 * 9 * 6, uci
    name_data(25)={'wdbc.txt'};           %569 * 30 * 2, uci
    name_data(26)={'breast.txt'};         %699 * 9 * 2
    name_data(27)={'leaves.txt'};         %1600 * 64 * 100, 36
    name_data(28)={'segment.txt'};        %2310 * 19 * 7, 38
    name_data(29)={'libras.txt'};         %360 * 90 * 15, 39
    name_data(30)={'ionosphere.txt'};     %351 * 34 * 2, 40
    name_data(31)={'waveform.txt'};       %5000 * 21 * 3, 41
    name_data(32)={'waveform_noise.txt'}; %5000 * 40 * 3, 42
    name_data(33)={'ecoli.txt'};          %336 * 7 * 8, 32
    name_data(34)={'cnae9.txt'};          %1080 * 856 * 9, 44
    name_data(35)={'Olivertti.txt'};      %400 * 28 * 40, 45
    name_data(36)={'dermatology.txt'};    %366 * 33 * 6, 46
    name_data(37)={'balance-scale.txt'};  %625 * 4 * 3, 47
    name_data(38)={'robotnavi.txt'};      %5456 * 24 * 4, 51
    name_data(39)={'scc.txt'};            %600 * 60 * 6, 52
    name_data(40)={'usps.txt'};           %11000 * 256 * 10, 54

    if flag_tsne==0
        direc='trued\';
        fname=[direc,name_data{idx}];
        descr=dlmread(fname);
        dimen=size(descr,2);
        label=descr(:,dimen);
        descr=descr(:,1:dimen-1);
    else
        direc='2d\';
        if idx<10 || idx==16
            fname=[direc,name_data{idx}];
        else
            sname=name_data{idx};
            sname=sname(1:length(sname)-4);
            sname=[sname,'-2d-tsne.txt'];
            fname=[direc,sname];
        end
        descr=dlmread(fname);
        dimen=size(descr,2);
        label=descr(:,dimen);
        descr=descr(:,1:dimen-1);
    end

end

%This function is used to extract a dominant set from a similarity matrix
%A, with x as the initial state, and toll as the error limit
%from S.R. Bulo, M. Pelillo, I.M. Bomze, Graph-based quadratic optimization: a
%fast evolutionary approach, Comput. Vis. Image Understand. 115 (7) (2011)
%984¨C995 .
%written by S. R. Bulo
function x=indyn(sima,x,toll)

    dsima=size(sima,1);
    if (~exist('x','var'))
        x=zeros(dsima,1);
        maxv=max(sima);
        for i=1:dsima
            if maxv(i)>0
                x(i)=1;
                break;
            end
        end
    end
    
    if (~exist('toll','var'))
        toll=0.005;
    end
    
    for i=1:dsima
        sima(i,i)=0;
    end
    
    x=reshape(x,dsima,1);

    %start operation
    g = sima*x;
    AT = sima;
    h = AT*x;
    niter=0;
    while 1,
        r = g - (x'*g);
        
        if norm(min(x,-r))<toll
            break;
        end
        i = selectPureStrategy(x,r);
        den = sima(i,i) - h(i) - r(i); %In case of asymmetric affinities
        do_remove=0;
        if r(i)>=0
            mu = 1;
            if den<0
                mu = min(mu, -r(i)/den);
                if mu<0 
                    mu=0; 
                end
            end
        else
            do_remove=1;
            mu = x(i)/(x(i)-1);
            if den<0
                [mu do_remove] = max([mu -r(i)/den]);
                do_remove=do_remove==1;
            end
        end
        tmp = -x;
        tmp(i) = tmp(i)+1;
        x = mu*tmp + x;
        if(do_remove) 
           x(i)=0; 
        end;
        x=abs(x)/sum(abs(x));
        
        g = mu*(sima(:,i)-g) + g;
        h = mu*(AT(:,i)-h) + h; %In case of asymmetric affinities
        niter=niter+1;
    end
    
    x=x';
end

function [i] = selectPureStrategy(x,r)
    index=1:length(x);
    mask = x>0;
    masked_index = index(mask);
    [~, i] = max(r);
    [~, j] = min(r(x>0));
    j = masked_index(j);
    if r(i)<-r(j)
        i = j;
    end
    return;
end

function z = nmi(x, y)
% Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length 
% Ouput:
%   z: normalized mutual information z=I(x,y)/sqrt(H(x)*H(y))
% Written by Mo Chen (sth4nth@gmail.com).
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);

l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));

idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log2(Pxy));

% hacking, to elimative the 0log0 issue
Px = nonzeros(mean(Mx,1));
Py = nonzeros(mean(My,1));

% entropy of Py and Px
Hx = -dot(Px,log2(Px));
Hy = -dot(Py,log2(Py));

% mutual information
MI = Hx + Hy - Hxy;

% normalized mutual information
z = sqrt((MI/Hx)*(MI/Hy));
z = max(0,z);

end

function rate=label2accuracy(label_c,label_t)

    ndata=length(label_c);
    matr_c=zeros(ndata);
    matr_t=zeros(ndata);
    
    nsame=0;
    for i=1:ndata
        for j=1:ndata
            if label_c(i)==label_c(j)
                matr_c(i,j)=1;
            end
            if label_t(i)==label_t(j)
                matr_t(i,j)=1;
            end
            
            if matr_c(i,j)==matr_t(i,j)
                nsame=nsame+1;
            end
        end
    end
    
    rate=nsame/(ndata*ndata);

end

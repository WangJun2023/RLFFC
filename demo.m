clear all
clc
dataset_names = {'Indian_Pines', 'Salinas', 'KSC', 'Botswana'};
classifier_names = {'KNN', 'SVM', 'LDA'};
svm_para = {'-c 5000.000000 -g 0.500000 -m 500 -t 2 -q',...
    '-c 100 -g 16 -m 500 -t 2 -q',...
    '-c 10000.000000 -g 16.000000 -m 500 -t 2 -q',...
    '-c 10000 -g 0.5 -m 500 -t 2 -q',...
    };
Dataset = get_data(dataset_names{1});
Dataset.train_ratio = 0.1;
Dataset.svm_para = svm_para{1, 1};
load('Indian_Pines-40.mat');

H = LP(X1); % calculate the laplacian matrix of each segmented region

d = 5; % the latent feature dimension

clusternum = 10; % the number of selected bands

view_num = size(H,3);
num = size(H,1);
L_ba = zeros(num,num);

for i = 1 : view_num
    L_ba = L_ba + (1 / view_num) * H(:,:,i);
end

H0 = zeros(num,num);

opt.disp = 0;
for p=1:view_num
    H(:,:,p) = (H(:,:,p)+H(:,:,p)')/2;
    [Hp, ~] = eigs(H(:,:,p), d, 'la', opt);
    H0 = H0 + (1/view_num)*H(:,:,p);
    HP(:,:,p) = Hp;
end
[Y_ba,~] = eigs(H0, d, 'la', opt);
lambda = 2.^-15;
beta = 0.1;
%lambda = 2.^[-15:1:15];
%beta = 0.01:0.01:0.1;

[F,~,~,~] = RLFFC(HP,d,lambda,Y_ba,beta,L_ba);
for k = 1 : length(clusternum)
    [IDX, C, SUMD, D] = kmeans(F,clusternum(k),'maxiter',100,'replicates',50,'emptyaction','singleton');
    [~,I] = min(D); % I: the selected band subset
end
[acc,~] = test_bs_accu(I, Dataset, 'KNN');


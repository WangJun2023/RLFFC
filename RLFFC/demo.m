clear all
clc
warning off;

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
